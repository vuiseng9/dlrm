
# Wrapping DLRM with NNCF
1. DLRM forward signature
    
    `dlrm(X, lS_o, lS_i)`

1. Create config with multiple inputs' info

1. Train loader returns

    `for j, (X, lS_o, lS_i, T) in enumerate(train_ld):`

1. As we register the train loader above for range initialization, NNCF complains that the expectation of data loader must be tuple of 2 (see line 76 of initialization.py)

1. To overcome the error, data loader output signature is modified from `(X, lS_o, lS_i, T)` to `((X, lS_o, lS_i), T)`

1. The implication of the data loader change is to 
    * (1) either adapt the existing train/validaton loop for the new format of loader's output
    * (2) or a change the signature of DLRM forward to accept tuple of two.

1. When i tried (1), it failed internally at range initilization because data loader output doesnt match dlrm forward signature. So (2) is inevitable. 
1. When i tried (2), it breaks the graph building function. I suspect that list of input_info are not contained as one? or it gets unpacked explicitly?

# Solution
Use InitializingDataLoader to wrap original data loader and translate the structure of data to the expected tuple of data and target
```python
class DLRMInitializingDataLoader(InitializingDataLoader):
            def get_inputs(self, dataloader_output):
                return (dataloader_output[0:3]), dict()
        
        initializing_train_ld = DLRMInitializingDataLoader(train_ld)

        config.nncf_config = register_default_init_args(config.nncf_config, initializing_train_ld, device=device)
````

# Changes required in NNCF to quantize DLRM
`nn.EmbeddingBag` is not supported by NNCF. They are missing in NNCFGraph and excluded for quantization. 

1. Add following in `operator_metatypes.py`, nodes of embedding bag will be visible in NNCF 
```python
@OPERATOR_METATYPES.register()
class EmbeddingBagMetatype(OperatorMetatype):
    name = "embedding_bag"
    torch_nn_functional_patch_spec = PatchSpec([name])
```

2. Add following in `nncf/layers.py to have embedding bag registered for weight quantization
```python
class NNCFEmbeddingBag(_NNCFModuleMixin, nn.EmbeddingBag):
    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.EmbeddingBag.__name__

        args = [module.num_embeddings, module.embedding_dim,
                module.max_norm, module.norm_type, module.scale_grad_by_freq,
                module.mode, module.sparse, module.weight,
                module.include_last_offset]
        nncf_embedding_bag = NNCFEmbeddingBag(*args)
        dict_update(nncf_embedding_bag.__dict__, module.__dict__)
        return nncf_embedding_bag
.
.
.
NNCF_MODULES_DICT = {
    NNCFConv1d: nn.Conv1d,
    NNCFConv2d: nn.Conv2d,
    NNCFConv3d: nn.Conv3d,
    NNCFLinear: nn.Linear,
    NNCFConvTranspose2d: nn.ConvTranspose2d,
    NNCFConvTranspose3d: nn.ConvTranspose3d,
    NNCFEmbedding: nn.Embedding,
    NNCFEmbeddingBag: nn.EmbeddingBag,
}
```