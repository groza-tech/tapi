import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Union, Tuple

class MarkovChainGenerator:
    def __init__(self):
        self.transitions = {
            'START': ['mod', 'trans', 'proc', 'hand'],
            'mod': ['ify', 'ulate', 'el'],
            'trans': ['form', 'late', 'mit'],
            'proc': ['ess', 'edure', 'eed'],
            'hand': ['le', 'ler']
        }
    
    def generate_word(self):
        current = 'START'
        result = ''
        while current in self.transitions:
            next_part = random.choice(self.transitions[current])
            result += next_part
            current = next_part
        return result

class TriggerOptimizer:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vocab = list(tokenizer.get_vocab().keys())[:500]
        self.markov_generator = MarkovChainGenerator()

    def generate_initial_trigger(self, method: str = 'markov', length: int = 20) -> List[str]:
        """Generate initial trigger using specified method."""
        if method == 'markov':
            return [
                '#' + self.markov_generator.generate_word(),
                self.markov_generator.generate_word(),
                self.markov_generator.generate_word()
            ]
        elif method == 'uniform':
            return ['x'] * length
        else:
            raise ValueError(f"Unknown method: {method}")

    def optimize_trigger(self, source_code: str, target_code: str, 
                        method: str = 'both', k: int = 3, max_iter: int = 15) -> List[str]:
        """Optimize trigger tokens using specified method(s)."""
        best_trigger = None
        best_loss = float('inf')

        methods_to_try = ['markov', 'uniform'] if method == 'both' else [method]

        for current_method in methods_to_try:
            print(f"\nTrying method: {current_method}")
            
            if current_method == 'markov':
                # Пробуем несколько марковских триггеров
                for _ in range(5):
                    initial_trigger = self.generate_initial_trigger(method='markov')
                    trigger, loss = self._optimize_single_trigger(
                        initial_trigger, source_code, target_code, k, max_iter
                    )
                    if loss < best_loss:
                        best_loss = loss
                        best_trigger = trigger
            else:
                # Пробуем uniform триггер
                initial_trigger = self.generate_initial_trigger(method='uniform')
                trigger, loss = self._optimize_single_trigger(
                    initial_trigger, source_code, target_code, k, max_iter
                )
                if loss < best_loss:
                    best_loss = loss
                    best_trigger = trigger

        return best_trigger

    def _optimize_single_trigger(self, trigger: List[str], source_code: str, 
                               target_code: str, k: int, max_iter: int) -> Tuple[List[str], float]:
        """Optimize a single trigger."""
        current_trigger = trigger.copy()
        best_trigger = trigger.copy()
        best_loss = float('inf')

        for iteration in range(max_iter):
            current_loss = self.get_loss(source_code, target_code, current_trigger)
            print(f"Iteration {iteration}, Loss: {current_loss:.4f}")
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_trigger = current_trigger.copy()

            if not self._update_trigger(current_trigger, source_code, target_code, k, current_loss):
                self._apply_random_perturbation(current_trigger)

        return best_trigger, best_loss
      
    def get_loss(self, source_code: str, target_code: str, trigger: Union[List[str], str], h: int = 2) -> float:
        """Calculate loss for given source code, target code and trigger."""
        trigger_text = " ".join(trigger) if isinstance(trigger, list) else trigger
        
        prompt = self._create_prompt(trigger_text, source_code, target_code)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        target_tokens = self.tokenizer(target_code, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            base_loss = self._compute_base_loss(logits, inputs)
            adversarial_loss = self._compute_adversarial_loss(logits, target_tokens, h)
            
            return ((base_loss + adversarial_loss) / (h + 1)).item()

    def _compute_base_loss(self, logits: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Compute base loss using cross entropy."""
        return F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            inputs.input_ids[:, 1:].reshape(-1)
        )

    def _compute_adversarial_loss(self, logits: torch.Tensor, target_tokens: torch.Tensor, h: int) -> torch.Tensor:
        """Compute adversarial loss component."""
        loss_e = 0
        for i in range(1, h+1):
            for j in range(1, i+1):
                token_logits = logits[:, j-1, :]
                target_token = target_tokens.input_ids[:, j-1]
                probs = F.softmax(token_logits, dim=-1)
                token_loss = -torch.log(1 - probs.gather(1, target_token.unsqueeze(1)))
                loss_e += token_loss.mean()
        return loss_e

    def _update_trigger(self, trigger: List[str], source_code: str, target_code: str, 
                       k: int, current_loss: float) -> bool:
        """Update trigger tokens based on gradients."""
        gradient = self._compute_gradient(source_code, target_code, trigger)
        indices = np.argsort(gradient)[-k:]
        updated = False
        
        for idx in indices:
            candidate_tokens = random.sample(self.vocab, min(20, len(self.vocab)))
            for new_token in candidate_tokens:
                temp_trigger = trigger.copy()
                temp_trigger[idx] = new_token
                new_loss = self.get_loss(source_code, target_code, temp_trigger)
                
                if new_loss < current_loss:
                    trigger[idx] = new_token
                    current_loss = new_loss
                    updated = True
                    break
        
        return updated

    def _compute_gradient(self, source_code: str, target_code: str, trigger: List[str]) -> np.ndarray:
        """Compute gradient for trigger tokens."""
        gradients = []
        base_loss = self.get_loss(source_code, target_code, trigger)
        
        for i in range(len(trigger)):
            token_gradients = []
            current_token = trigger[i]
            
            candidate_tokens = random.sample(self.vocab, min(5, len(self.vocab)))
            for candidate_token in candidate_tokens:
                if candidate_token != current_token:
                    temp_trigger = trigger.copy()
                    temp_trigger[i] = candidate_token
                    new_loss = self.get_loss(source_code, target_code, temp_trigger)
                    token_gradients.append(new_loss - base_loss)
            
            gradients.append(np.mean(token_gradients) if token_gradients else 0)
        
        return np.array(gradients)

    def _apply_random_perturbation(self, trigger: List[str]):
        """Apply random perturbation to trigger."""
        if random.random() < 0.3:  # 30% chance of random change
            random_idx = random.randint(0, len(trigger)-1)
            trigger[random_idx] = random.choice(self.vocab)


    def _create_prompt(self, trigger_text: str, source_code: str, target_code: str) -> str:
        return f"""<PRE>
{trigger_text}

{source_code}
# Replace with:
{target_code}
</PRE>"""
