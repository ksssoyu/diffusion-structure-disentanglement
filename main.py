import argparse
import os
import torch
import gc
from diffusers import StableDiffusionPipeline
from PIL import Image

from src.processor import GradualInjectionProcessor
from src.utils import increment_step_count, reset_step_count

def parse_args():
    parser = argparse.ArgumentParser(description="Structure-Aware Editing with Attention Injection")
    
    # 기본 설정
    parser.add_argument("--prompt_a", type=str, default="A photo of a cute cat looking at the camera, highly detailed", help="Source Image Prompt")
    parser.add_argument("--prompt_b", type=str, default="A pixel art character of a cat, 8-bit style", help="Target Image Prompt")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps")
    parser.add_argument("--threshold", type=float, default=0.8, help="Injection threshold (0.0 - 1.0)")
    
    # 실험 모드 선택
    parser.add_argument("--experiment_type", type=str, default="attention", choices=["layer", "attention"], 
                        help="Choose experiment type: 'layer' (Up/Down blocks) or 'attention' (Self/Cross attn)")
    
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 설정 및 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading Stable Diffusion (Experiment: {args.experiment_type})...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None
    
    # 2. Source 생성
    print(f"[Step 1] Generating Source: '{args.prompt_a}'")
    stored_maps = []
    
    # 모든 어텐션 레이어에 리스너 부착
    for name, module in pipe.unet.named_modules():
        if ("attn1" in name or "attn2" in name) and hasattr(module, "processor"):
            module.processor = GradualInjectionProcessor(
                store_controller=stored_maps,
                layer_name=name
            )
            
    generator = torch.Generator(device).manual_seed(args.seed)
    reset_step_count(pipe)
    image_a = pipe(args.prompt_a, generator=generator, num_inference_steps=args.steps, 
                   callback=lambda s, t, l: increment_step_count(pipe) or l, callback_steps=1).images[0]
    
    image_a.save(os.path.join(args.output_dir, "source_A.png"))
    print(" -> Source saved.")

    # 3. 실험 조건 설정
    if args.experiment_type == "layer":
        # 레이어 위치별 실험
        conditions = [
            ("all", "All Layers"),
            ("down", "Down Blocks"),
            ("mid", "Mid Block"),
            ("up", "Up Blocks")
        ]
    else: # attention
        # 어텐션 종류별 실험 (Self vs Cross)
        conditions = [
            ("all", "All Attn"),
            ("self", "Self Attn Only"),
            ("cross", "Cross Attn Only")
        ]

    # 4. Target 생성 루프
    print(f"[Step 2] Generating Targets: '{args.prompt_b}'")
    
    for cond_key, label in conditions:
        print(f" -> Processing: {label}...")
        
        # 맵 복사
        maps_to_inject = [m.clone() for m in stored_maps]
        
        for name, module in pipe.unet.named_modules():
            if ("attn1" in name or "attn2" in name) and hasattr(module, "processor"):
                
                # 주입 여부 판단 로직
                should_inject_layer = False
                target_mode = "all" # 기본값

                if args.experiment_type == "layer":
                    # 레이어 위치 기반 필터링
                    if cond_key == "all": should_inject_layer = True
                    elif cond_key == "down" and "down_blocks" in name: should_inject_layer = True
                    elif cond_key == "mid" and "mid_block" in name: should_inject_layer = True
                    elif cond_key == "up" and "up_blocks" in name: should_inject_layer = True
                    
                    # 레이어 실험에서는 threshold를 0으로 만드는 방식으로 On/Off 제어
                    current_threshold = args.threshold if should_inject_layer else 0.0
                    
                else: # attention type 실험
                    # 어텐션 타입 기반 필터링 (processor 내부 mode 파라미터 사용)
                    current_threshold = args.threshold
                    target_mode = cond_key # 'self', 'cross', 'all' 전달

                # Processor 부착
                module.processor = GradualInjectionProcessor(
                    inject_controller=maps_to_inject,
                    threshold=current_threshold,
                    layer_name=name,
                    mode=target_mode
                )

        # 생성 수행
        generator = torch.Generator(device).manual_seed(args.seed)
        reset_step_count(pipe)
        
        try:
            img = pipe(args.prompt_b, generator=generator, num_inference_steps=args.steps, 
                       callback=lambda s, t, l: increment_step_count(pipe) or l, callback_steps=1).images[0]
            
            # 파일명 저장
            filename = f"result_{args.experiment_type}_{cond_key}.png"
            img.save(os.path.join(args.output_dir, filename))
            print(f"    Saved: {filename}")
            
        except Exception as e:
            print(f"    Error: {e}")
            
        # 메모리 정리
        del maps_to_inject
        gc.collect()
        torch.cuda.empty_cache()

    print("Done! Check the output directory.")

if __name__ == "__main__":
    main()