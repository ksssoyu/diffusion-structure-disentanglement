import torch

class GradualInjectionProcessor:
    def __init__(self, store_controller=None, inject_controller=None, 
                 threshold=0.0, layer_name="", mode="all"):
        """
        Args:
            store_controller (list): Attention Map을 저장할 리스트 (Source 단계)
            inject_controller (list): 주입할 Attention Map이 담긴 리스트 (Target 단계)
            threshold (float): 주입 강도 (0.0 ~ 1.0)
            layer_name (str): 현재 레이어의 이름 (예: "down_blocks.1.attn1")
            mode (str): 주입 모드 ("all", "self", "cross")
        """
        self.store_controller = store_controller   
        self.inject_controller = inject_controller 
        self.threshold = threshold                 
        self.layer_name = layer_name
        self.mode = mode
        self.step_count = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        # --- [기본 연산 준비] ---
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None: encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # ========================================================
        # [메모리 최적화 및 제어 로직]
        # ========================================================
        
        # 64x64 (seq_len=4096) 처럼 너무 큰 맵은 메모리 문제로 건너뜀
        # Colab RAM 보호를 위해 32x32 (seq_len=1024) 이하만 처리
        is_too_large = (sequence_length > 1024) 

        if not is_too_large:
            # 1) 저장 모드 (Source)
            if self.store_controller is not None:
                # float16으로 줄여서 저장 (용량 절약)
                self.store_controller.append(attention_probs.detach().cpu().to(torch.float16))
                
            # 2) 주입 모드 (Target)
            if self.inject_controller is not None and len(self.inject_controller) > 0:
                # 무조건 꺼내기 (Sync 맞추기 위해 pop은 필수)
                ref_attn = self.inject_controller.pop(0).to(hidden_states.device)
                
                # 주입 조건 검사
                should_inject = False
                if self.mode == "all": should_inject = True
                elif self.mode == "self" and "attn1" in self.layer_name: should_inject = True
                elif self.mode == "cross" and "attn2" in self.layer_name: should_inject = True
                    
                # 스텝 조건 검사
                total_steps = 20 # Inference Step 수와 맞춰야 함
                if should_inject and (self.step_count < (total_steps * self.threshold)):
                    attention_probs = ref_attn

        # ========================================================

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states