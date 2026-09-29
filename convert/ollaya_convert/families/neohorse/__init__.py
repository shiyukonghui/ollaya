"""NeoHorse-Jev-4B: a Qwen3.5-4B multimodal backbone with an independent decision pointer head.

A *decision* model (it never generates): the readout is `k(h[opt]) . q(h[decide])` over the
backbone's final hidden state, the same primitive as `kev-pointer-v1`, on the multimodal backbone
that also reads an image (`decider-vision-v1`'s vision tower). This family composes the two.
"""
