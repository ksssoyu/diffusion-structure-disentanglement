import torch

class GradualInjectionProcessor:
    def __init__(self, store_controller=None, inject_controller=None, 
                 start_ratio=0.0, end_ratio=0.0,
                 layer_name="", mode="all"):

        self.store_controller = store_controller   
        self.inject_controller = inject_controller 
        self.start_ratio = start_ratio       
        self.end_ratio = end_ratio     
        self.layer_name = layer_name
        self.mode = mode
        self.step_count = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        # --- 기본 연산 준비 ---
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
                
        # 64x64 (seq_len=4096) 이상은 메모리 절약을 위해 건너뜀
        is_too_large = (sequence_length > 1024) 

        if not is_too_large:
            # 1) 저장 모드
            if self.store_controller is not None:
                self.store_controller.append(attention_probs.detach().cpu().to(torch.float16))
                
            # 2) 주입 모드
            if self.inject_controller is not None and len(self.inject_controller) > 0:
                ref_attn = self.inject_controller.pop(0).to(hidden_states.device)
                
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
                    attention_probs = ref_attn

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
