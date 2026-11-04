# r-and-d-foundations

A curated repository of foundational research, hardware-level execution prototypes, and low-level C++ and CUDA kernels.
#import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTierDomeNetwork(nn.Module):
    def __init__(self, d_model=512, buffer_size=64, num_slots=32, archive_size=16):
        super().__init__()
        self.d_model = d_model
        
        # Level 1: Instant exact preservation buffer (O(1) RAM footprint)
        self.register_buffer("short_term_buffer", torch.zeros(buffer_size, d_model))
        
        # Level 2: Resonant memory slots (medium-term context)
        self.slots = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        
        # ==========================================
        # LEVEL 3: Archive Collector (Long-term Manifold)
        # ==========================================
        # Stores global patterns as compact fixed-size vectors
        self.archive_manifold = nn.Parameter(torch.randn(archive_size, d_model) * 0.01)
        
        # Phase coupling matrices for tunneling into the archive
        self.archive_gate = nn.Linear(d_model * 2, 1)
        self.gate_w = nn.Linear(d_model * 2, 1)

    def forward(self, x_t):
        """
        x_t: current incoming token [1, d_model]
        """
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
            
        batch_size = x_t.size(0)

        # 1. Level 1: Sliding buffer
        self.short_term_buffer = torch.cat([self.short_term_buffer[batch_size:, :], x_t.detach()], dim=0)
        buffer_scores = torch.matmul(x_t, self.short_term_buffer.transpose(0, 1)) / (self.d_model ** 0.5)
        buffer_context = torch.matmul(F.softmax(buffer_scores, dim=-1), self.short_term_buffer)

        # 2. Level 2: Resonant slots
        slot_scores = torch.matmul(x_t, self.slots.transpose(0, 1)) / (self.d_model ** 0.5)
        slot_weights = F.softmax(slot_scores, dim=-1)
        slot_context = torch.matmul(slot_weights, self.slots)

        # ==========================================
        # 3. Level 3 Operation (Archive Resonance / Tunneling)
        # ==========================================
        # Check if the current input resonates with the global archive manifold
        archive_scores = torch.matmul(x_t, self.archive_manifold.transpose(0, 1)) / (self.d_model ** 0.5)
        archive_weights = F.softmax(archive_scores, dim=-1)
        archive_context = torch.matmul(archive_weights, self.archive_manifold) # [batch_size, d_model]

        # If phases/meanings match, the archive "tunnels" information directly into the current circuit
        resonance_gate = torch.sigmoid(self.archive_gate(torch.cat([x_t, archive_context], dim=-1)))
        deep_memory_pull = resonance_gate * archive_context

        with torch.no_grad():
            # Slow background drift of the archive: the most stable patterns of Level 2 
            # are gradually conserved into the archive manifold, replacing old obsolete frequencies
            self.archive_manifold.data = 0.999 * self.archive_manifold.data + 0.001 * slot_context.mean(dim=0, keepdim=True)

        # Synthesis of the final representation with linear complexity O(N)
        combined_output = x_t + buffer_context + slot_context + deep_memory_pull
        
        return combined_output

# --- Three-tier integration test ---
if __name__ == "__main__":
    d_model = 256
    model = MultiTierDomeNetwork(d_model=d_model, buffer_size=32, num_slots=16, archive_size=8)
    
    print("Initialized three-tier memory architecture...")
    for step in range(500):
        token = torch.randn(1, d_model)
        out = model(token)
        
    print("500 steps completed successfully.")
    print("Memory footprint (fixed sizes):")
    print(" - Level 1 (Buffer):", model.short_term_buffer.shape)
    print(" - Level 2 (Slots):", model.slots.shape)
    print(" - Level 3 (Archive):", model.archive_manifold.shape) (Archive)

A curated repository of foundational research, hardware-level execution prototypes, and low-level C++/CUDA kernels. 

### Content Structure
- `/cuda-kernels` — Early memory-layout tests and tensor execution primitives (SoA).
- `/bare-metal` — Low-level memory management and hardware-level prototypes.
- `/physical-computing` — Mathematical specifications, particle dynamics, and scattering optics models.

> **Note:** This repository serves as an archival record of experimental research that laid the architectural groundwork for the AS3D and ERE projects.
> import torch
import torch.nn as nn
import torch.nn.functional as F

class DomeAttentionCell(nn.Module):
    def __init__(self, d_model=512, buffer_size=128, num_slots=64):
        super().__init__()
        self.d_model = d_model
        self.buffer_size = buffer_size   # Level 1: exact preservation window size
        self.num_slots = num_slots       # Level 2: fixed number of memory slots
        
        # Level 1: Sliding buffer of exact vectors (stored in fast memory)
        # Size: [buffer_size, d_model] -> at d_model=512 and 128 tokens, this is a fraction of a megabyte
        self.register_buffer("short_term_buffer", torch.zeros(buffer_size, d_model))
        
        # Level 2: Resonant memory slots (state matrix replacing old history)
        # Size: [num_slots, d_model] -> fixed RAM footprint O(1)
        self.slots = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        
        # Dynamic update gates and phase coupling
        self.gate_w = nn.Linear(d_model * 2, 1)
        self.phase_proj = nn.Linear(d_model, d_model) # For emulating phase resonance

    def forward(self, x_t):
        """
        x_t: vector of the current incoming token, size [batch_size, d_model] or [d_model]
        """
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0) # Ensure batch dimension exists [1, d_model]
            
        batch_size = x_t.size(0)

        # ==========================================
        # LEVEL 1: Instant Exact-Preservation Buffer
        # ==========================================
        # Shift the buffer down and write the new token at the top (sliding window)
        self.short_term_buffer = torch.cat([self.short_term_buffer[batch_size:, :], x_t.detach()], dim=0)
        
        # Compute local attention inside the buffer (O(window_size^2), but window is strictly fixed)
        # This provides ideal precision for the last N tokens
        buffer_scores = torch.matmul(x_t, self.short_term_buffer.transpose(0, 1)) / (self.d_model ** 0.5)
        buffer_weights = F.softmax(buffer_scores, dim=-1)
        buffer_context = torch.matmul(buffer_weights, self.short_term_buffer) # [batch_size, d_model]

        # ==========================================
        # LEVEL 2: Resonant Core (Memory Slots)
        # ==========================================
        # Emulation of phase coupling (dot product acting as a trigonometric phase filter)
        slot_scores = torch.matmul(x_t, self.slots.transpose(0, 1)) / (self.d_model ** 0.5)
        slot_weights = F.softmax(slot_scores, dim=-1) # [batch_size, num_slots]
        slot_context = torch.matmul(slot_weights, self.slots) # [batch_size, d_model]

        # Soft update of slots (information from Level 1 "settles" into Level 2 upon overflow)
        update_gate = torch.sigmoid(self.gate_w(torch.cat([x_t, slot_context], dim=-1)))
        
        with torch.no_grad():
            # Slow drift of slots: old meanings are smoothly updated with new trends
            self.slots.data = (1 - update_gate.mean()) * self.slots.data + update_gate.mean() * x_t.repeat(self.num_slots, 1).mean(dim=0, keepdim=True)

        # ==========================================
        # SYNTHESIS (Core Output with Linear Complexity)
        # ==========================================
        # Combine the exact locality of Level 1 and the compressed "infinite" memory of Level 2
        combined_output = x_t + buffer_context + slot_context
        
        return combined_output

# --- Verification check example ---
if __name__ == "__main__":
    d_model = 256
    model = DomeAttentionCell(d_model=d_model, buffer_size=64, num_slots=32)
    
    # Simulate a stream of 1000 steps (context grows, but RAM remains constant!)
    print("Running data stream with fixed RAM footprint...")
    for step in range(1000):
        dummy_token = torch.randn(1, d_model)
        output = model(dummy_token)
        #######
        import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTierDomeNetwork(nn.Module):
    def __init__(self, d_model=512, buffer_size=64, num_slots=32, archive_size=16):
        super().__init__()
        self.d_model = d_model
        
        # Level 1: Instant exact preservation buffer (O(1) RAM footprint)
        self.register_buffer("short_term_buffer", torch.zeros(buffer_size, d_model))
        
        # Level 2: Resonant memory slots (medium-term context)
        self.slots = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        
        # ==========================================
        # LEVEL 3: Archive Collector (Long-term Manifold)
        # ==========================================
        # Stores global patterns as compact fixed-size vectors
        self.archive_manifold = nn.Parameter(torch.randn(archive_size, d_model) * 0.01)
        
        # Phase coupling matrices for tunneling into the archive
        self.archive_gate = nn.Linear(d_model * 2, 1)
        self.gate_w = nn.Linear(d_model * 2, 1)

    def forward(self, x_t):
        """
        x_t: current incoming token [1, d_model]
        """
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
            
        batch_size = x_t.size(0)

        # 1. Level 1: Sliding buffer
        self.short_term_buffer = torch.cat([self.short_term_buffer[batch_size:, :], x_t.detach()], dim=0)
        buffer_scores = torch.matmul(x_t, self.short_term_buffer.transpose(0, 1)) / (self.d_model ** 0.5)
        buffer_context = torch.matmul(F.softmax(buffer_scores, dim=-1), self.short_term_buffer)

        # 2. Level 2: Resonant slots
        slot_scores = torch.matmul(x_t, self.slots.transpose(0, 1)) / (self.d_model ** 0.5)
        slot_weights = F.softmax(slot_scores, dim=-1)
        slot_context = torch.matmul(slot_weights, self.slots)

        # ==========================================
        # 3. Level 3 Operation (Archive Resonance / Tunneling)
        # ==========================================
        # Check if the current input resonates with the global archive manifold
        archive_scores = torch.matmul(x_t, self.archive_manifold.transpose(0, 1)) / (self.d_model ** 0.5)
        archive_weights = F.softmax(archive_scores, dim=-1)
        archive_context = torch.matmul(archive_weights, self.archive_manifold) # [batch_size, d_model]

        # If phases/meanings match, the archive "tunnels" information directly into the current circuit
        resonance_gate = torch.sigmoid(self.archive_gate(torch.cat([x_t, archive_context], dim=-1)))
        deep_memory_pull = resonance_gate * archive_context

        with torch.no_grad():
            # Slow background drift of the archive: the most stable patterns of Level 2 
            # are gradually conserved into the archive manifold, replacing old obsolete frequencies
            self.archive_manifold.data = 0.999 * self.archive_manifold.data + 0.001 * slot_context.mean(dim=0, keepdim=True)

        # Synthesis of the final representation with linear complexity O(N)
        combined_output = x_t + buffer_context + slot_context + deep_memory_pull
        
        return combined_output

# --- Three-tier integration test ---
if __name__ == "__main__":
    d_model = 256
    model = MultiTierDomeNetwork(d_model=d_model, buffer_size=32, num_slots=16, archive_size=8)
    
    print("Initialized three-tier memory architecture...")
    for step in range(500):
        token = torch.randn(1, d_model)
        out = model(token)
        
    print("500 steps completed successfully.")
    print("Memory footprint (fixed sizes):")
    print(" - Level 1 (Buffer):", model.short_term_buffer.shape)
    print(" - Level 2 (Slots):", model.slots.shape)
    print(" - Level 3 (Archive):", model.archive_manifold.shape)
    #####
    print(1000, "tokens processed successfully.")
    print("Level 1 Memory Size (Buffer):", model.short_term_buffer.shape)
    print("Level 2 Memory Size (Slots):", model.slots.shape)
