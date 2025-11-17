import torch
from typing import Optional, List

class GradualInjectionProcessor:
    def __init__(self, store_controller=None, inject_controller=None, 
                 start_ratio=0.0, end_ratio=0.0,
                 layer_name="", mode="all",
                 inject_head_indices: Optional[List[int]] = None):

        self.store_controller = store_controller    
        self.inject_controller = inject_controller  
        self.start_ratio = start_ratio      
        self.end_ratio = end_ratio      
        self.layer_name = layer_name
        self.mode = mode
        self.inject_head_indices = inject_head_indices
        self.step_count = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        # --- 기본 연산 준비 ---
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None: encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        num_heads = attn.heads

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # target_map: 현재 스텝에서 계산된 원본 어텐션 맵
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
                
        is_too_large = (sequence_length > 1024) 

        if not is_too_large:
            # 1) 저장 모드
            if self.store_controller is not None:
                self.store_controller.append(attention_probs.detach().cpu().to(torch.float16))
                
            # 2) 주입 모드
            if self.inject_controller is not None and len(self.inject_controller) > 0:
                # source_map_flat: 주입할 소스 어텐션 맵
                ref_attn_flat = self.inject_controller.pop(0).to(hidden_states.device)
                
                # 주입 모드 체크
                should_inject_type = False
                if self.mode == "all": should_inject_type = True
                elif self.mode == "self" and "attn1" in self.layer_name: should_inject_type = True
                elif self.mode == "cross" and "attn2" in self.layer_name: should_inject_type = True
                        
                # 구간(Temporal Range) 체크
                total_steps = 20
                current_ratio = self.step_count / total_steps
                
                is_in_time_range = (self.start_ratio <= current_ratio < self.end_ratio)

                if should_inject_type and is_in_time_range:
                    
                    if self.inject_head_indices is None:
                        # 인덱스가 None이면, 맵 전체를 교체
                        attention_probs = ref_attn_flat
                    
                    else:
                        
                        # 1. 맵을 (Batch, Heads, Seq_Len, Seq_Len)으로 Reshape
                        target_map = attention_probs.view(batch_size, num_heads, sequence_length, sequence_length)
                        source_map = ref_attn_flat.view(batch_size, num_heads, sequence_length, sequence_length)
                        
                        # 2. target_map을 복제하여 final_map을 만듦
                        final_map = target_map.clone()

                        # 3. 지정된 인덱스의 헤드만 source_map에서 복사
                        for h_idx in self.inject_head_indices:
                            if h_idx < num_heads: # 안전장치
                                final_map[:, h_idx, :, :] = source_map[:, h_idx, :, :]
                        
                        # 4. 'final_map'을 다시 원래의 flat 형태로 Reshape
                        # (b, h, s, s) -> (b*h, s, s)
                        attention_probs = final_map.view(batch_size * num_heads, sequence_length, sequence_length)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
