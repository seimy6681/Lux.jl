# Major parts of this file are adapted from the following sources:
#   1. https://github.com/FluxML/Flux.jl/blob/fa108bb994a2cc839240d8497c9e1610818a49ab/src/layers/attention.jl
"""
    MultiHeadAttention(dims; nheads=1, dense_kwargs=(; use_bias=False()),
                       attention_dropout_probability=0.0f0,
                       is_causal::Union{Bool,Nothing}=nothing)

The multi-head dot-product attention layer used in Transformer architectures
[vaswani2017attention](@citep).

## Arguments

  - `dims`: The embedding dimensions of inputs, intermediate tensors and outputs.
    In the most general case, it is given as

    + a) `(q_in_dim, k_in_dim, v_in_dim) => (qk_dim, v_dim) => out_dim`.

    Can take also simpler forms as

    + b) `dims::Int`;
    + c) `in_dim::Int => (qk_dim, v_dim) => out_dim`;
    + d) `in_dim::Int => qkv_dim => out_dim`.

## Keyword Arguments

  - `nheads`: number of heads.
  - `attention_dropout_probability`: dropout probability for the attention scores.
  - `dense_kwargs`: keyword arguments for the Dense layers. Default `use_bias=false`.
  - `is_causal`: whether the attention is causal. If this is provided, the attention mask
    will be automatically created (passing in the `mask` argument is not allowed and will
    throw an error).

## Forward Pass Signature(s)

```julia
(m::MultiHeadAttention)(qkv, ps, st::NamedTuple)
(m::MultiHeadAttention)((q, kv), ps, st::NamedTuple)
(m::MultiHeadAttention)((q, k, v, [mask = nothing]), ps, st::NamedTuple)
```

## Inputs

  - `qkv`: a single input tensor for query, key and value. This corresponds to
    self-attention.
  - `(q, kv)`: a tuple of two input tensors for query and key-value.
  - `(q, k, v)`: a tuple of three input tensors for query, key and value.
  - `mask`: an optional mask to apply to the attention scores. This must be broadcastable
    to the shape of the attention scores `(kv_len, q_len, nheads, batch_size)`.

The query tensor `q` is expected to have shape `(q_in_dim, q_len, batch_size)`, the
key test `k` is expected to have shape `(k_in_dim, kv_len, batch_size)`, the value
tensor `v` is expected to have shape `(v_in_dim, kv_len, batch_size)`.

## Returns

  - A tuple of two elements. The first element is the output tensor of shape
    `(out_dim, q_len, batch_size)` and the second element is the attention scores
    of shape `(q_len, kv_len, nheads, batch_size)`.
  - A NamedTuple of the states of the layer.

# Extended Help

## Examples

```jldoctest
julia> m = MultiHeadAttention(64; nheads=8);

julia> ps, st = Lux.setup(Random.default_rng(), m);

julia> q = randn(Float32, 64, 10, 32);

julia> k = randn(Float32, 64, 20, 32);

julia> v = randn(Float32, 64, 20, 32);

julia> (y, α), st_new = m((q, k, v), ps, st);

julia> size(y)
(64, 10, 32)

julia> size(α)
(20, 10, 8, 32)

julia> (y, α), st_new = m(q, ps, st);  # self-attention

julia> size(y)
(64, 10, 32)

julia> size(α)
(10, 10, 8, 32)
```

"""
@concrete struct MultiHeadAttention <: AbstractLuxContainerLayer{(
    :q_proj, :k_proj, :v_proj, :attention_dropout, :out_proj # specifiying which struct fields are lyaers to generate ps and st trees
),}
    nheads <: IntegerType # number of attention heads
    q_proj <: Dense # query projection layer (W_q)
    k_proj <: Dense # key projection layer (W_k)
    v_proj <: Dense # value projection layer (W_v)
    attention_dropout
    out_proj <: Dense # output projection layer (W_o)
    is_causal <: Union{Bool,Nothing} # causal mask flag 
end

function Base.show(io::IO, ::MIME"text/plain", mha::MultiHeadAttention)
    q_in, k_in, v_in = mha.q_proj.in_dims, mha.k_proj.in_dims, mha.v_proj.in_dims
    out_dim = mha.out_proj.out_dims

    attention_dropout_probability =
        mha.attention_dropout isa NoOpLayer ? 0.0f0 : mha.attention_dropout.p
    print(
        io, "MultiHeadAttention($q_in => ($k_in, $v_in) => $out_dim; nheads=$(mha.nheads)"
    )
    !iszero(attention_dropout_probability) &&
        print(io, ", attention_dropout_probability=$(attention_dropout_probability)")
    mha.is_causal !== nothing && print(io, ", is_causal=$(mha.is_causal)")
    print(io, ")")
    return nothing
end

function parse_mha_dims(dims::IntegerType)
    return (; q_in=dims, k_in=dims, v_in=dims, qk=dims, v=dims, out=dims)
end

function parse_mha_dims((in_dims, (qkv_dims, out_dims)))
    if in_dims isa IntegerType
        q_in, k_in, v_in = in_dims, in_dims, in_dims
    else
        @assert in_dims isa NTuple{3,<:IntegerType}
        q_in, k_in, v_in = in_dims
    end

    if qkv_dims isa IntegerType
        qk, v = qkv_dims, qkv_dims
    else
        @assert qkv_dims isa NTuple{2,<:IntegerType}
        qk, v = qkv_dims
    end

    return (; q_in, k_in, v_in, qk, v, out=out_dims)
end

function MultiHeadAttention(
    dims;
    nheads::IntegerType=1, # defaults to single head attention
    dense_kwargs=(; use_bias=False()),
    attention_dropout_probability=0.0f0,
    is_causal::Union{Bool,Nothing}=nothing,
)
    dims = parse_mha_dims(dims)
    @assert dims.qk % nheads == 0
    @assert dims.v % nheads == 0

    return MultiHeadAttention(
        nheads,
        Dense(dims.q_in, dims.qk; dense_kwargs...),
        Dense(dims.k_in, dims.qk; dense_kwargs...),
        Dense(dims.v_in, dims.v; dense_kwargs...),
        Dropout(attention_dropout_probability),
        Dense(dims.v, dims.out; dense_kwargs...),
        is_causal,
    )
end

@trace function (mha::MultiHeadAttention)(x, ps, st::NamedTuple)
    return apply_multiheadattention(mha, ps, st, x)
end
@trace function (mha::MultiHeadAttention)(x::Tuple, ps, st::NamedTuple)
    return apply_multiheadattention(mha, ps, st, x...)
end

function apply_multiheadattention(mha::MultiHeadAttention, ps, st, qkv)
    return apply_multiheadattention(mha, ps, st, qkv, qkv, qkv, nothing)
end

function apply_multiheadattention(mha::MultiHeadAttention, ps, st, q, kv)
    return apply_multiheadattention(mha, ps, st, q, kv, kv, nothing)
end

function apply_multiheadattention(mha::MultiHeadAttention, ps, st, q, k, v, mask=nothing)
    q, k, v = match_eltype(mha, ps, st, q, k, v) # ensuring q, k, v have the same eltype as layer's parameters

    # apply linear transformation to map raw inputs into query, key and value feature spaces
    q, q_st = mha.q_proj(q, ps.q_proj, st.q_proj)
    k, k_st = mha.k_proj(k, ps.k_proj, st.k_proj)
    v, v_st = mha.v_proj(v, ps.v_proj, st.v_proj)

    dropout = StatefulLuxLayer(
        mha.attention_dropout, ps.attention_dropout, st.attention_dropout
    ) # statefulLuxLayer avoids passing ps and st to the dropout layer

    x, α = scaled_dot_product_attention(
        reshape(q, size(q, 1) ÷ mha.nheads, mha.nheads, size(q)[2:end]...), # [q_out_dim, seq_len, batch] -> [head_dim, nheads, seq_len, batch]
        reshape(k, size(k, 1) ÷ mha.nheads, mha.nheads, size(k)[2:end]...),
        reshape(v, size(v, 1) ÷ mha.nheads, mha.nheads, size(v)[2:end]...);
        head_dim=1, # feature vectors per head are along the first dimension
        token_dim=3, # se_len (num tokens) dimension is along the 3rd dimension
        fdrop=dropout,
        mask,
        mha.is_causal,
    ) # x: output attention tensor, α: attention weights (softmax probabilities)
    x = reshape(x, size(x, 1) * mha.nheads, size(x)[3:end]...) # head_dim * nheads. [q_out_dim, seq_len, batch]

    y, out_st = mha.out_proj(x, ps.out_proj, st.out_proj)

    return (
        (y, α),
        (;
            q_proj=q_st,
            k_proj=k_st,
            v_proj=v_st,
            attention_dropout=dropout.st,
            out_proj=out_st,
        ),
    )
end



@concrete struct GroupQueryAttention <: AbstractLuxContainerLayer{(
    :q_proj, :k_proj, :v_proj, :attention_dropout, :out_proj
),}
    nqheads <: IntegerType
    nkvheads <: IntegerType
    q_proj <: Dense
    k_proj <: Dense
    v_proj <: Dense
    attention_dropout
    out_proj <: Dense
    is_causal <: Union{Bool,Nothing}
end

function GroupQueryAttention(
    dims;
    nqheads::IntegerType,
    nkvheads::IntegerType,
    dense_kwargs=(; use_bias=False()),
    attention_dropout_probability=0.0f0,
    is_causal::Union{Bool,Nothing}=nothing,
)
    @assert nqheads % nkvheads == 0 "nqheads ($nqheads) must be divisible by nkvheads ($nkvheads)"

    parsed_dims = parse_mha_dims(dims)
    @assert parsed_dims.qk % nqheads == 0 "qk_dim must be divisible by nqheads"
    @assert parsed_dims.v % nqheads == 0 "v_dim must be divisible by nqheads"

    # Per-head dimension calculations
    head_dim_qk = parsed_dims.qk ÷ nqheads
    head_dim_v  = parsed_dims.v ÷ nqheads

    # Key and Value target output dimensions
    k_out_dim = head_dim_qk * nkvheads
    v_out_dim = head_dim_v * nkvheads

    return GroupQueryAttention(
        nqheads,
        nkvheads,
        Dense(parsed_dims.q_in, parsed_dims.qk; dense_kwargs...),
        Dense(parsed_dims.k_in, k_out_dim; dense_kwargs...),
        Dense(parsed_dims.v_in, v_out_dim; dense_kwargs...),
        Dropout(attention_dropout_probability),
        Dense(parsed_dims.v, parsed_dims.out; dense_kwargs...), # Inputs from parsed_dims.v (head_dim_v * nqheads)
        is_causal,
    )
end

@trace function (gqa::GroupQueryAttention)(x, ps, st::NamedTuple)
    return apply_groupqueryattention(gqa, ps, st, x)
end

@trace function (gqa::GroupQueryAttention)(x::Tuple, ps, st::NamedTuple)
    return apply_groupqueryattention(gqa, ps, st, x...)
end

function apply_groupqueryattention(gqa::GroupQueryAttention, ps, st, qkv)
    return apply_groupqueryattention(gqa, ps, st, qkv, qkv, qkv, nothing)
end

function apply_groupqueryattention(gqa::GroupQueryAttention, ps, st, q, kv)
    return apply_groupqueryattention(gqa, ps, st, q, kv, kv, nothing)
end

function apply_groupqueryattention(gqa::GroupQueryAttention, ps, st, q, k, v, mask=nothing)
    q, k, v = match_eltype(gqa, ps, st, q, k, v)

    q, q_st = gqa.q_proj(q, ps.q_proj, st.q_proj)
    k, k_st = gqa.k_proj(k, ps.k_proj, st.k_proj)
    v, v_st = gqa.v_proj(v, ps.v_proj, st.v_proj)

    dropout = StatefulLuxLayer(
        gqa.attention_dropout, ps.attention_dropout, st.attention_dropout
    )

    x, α = scaled_dot_product_attention(
        reshape(q, size(q, 1) ÷ gqa.nqheads,  gqa.nqheads,  size(q)[2:end]...),
        reshape(k, size(k, 1) ÷ gqa.nkvheads, gqa.nkvheads, size(k)[2:end]...), # Uses nkvheads
        reshape(v, size(v, 1) ÷ gqa.nkvheads, gqa.nkvheads, size(v)[2:end]...); # Uses nkvheads
        head_dim=1,
        token_dim=3,
        fdrop=dropout,
        mask,
        gqa.is_causal,
    )

    x = reshape(x, size(x, 1) * gqa.nqheads, size(x)[3:end]...)

    y, out_st = gqa.out_proj(x, ps.out_proj, st.out_proj)

    return (
        (y, α),
        (;
            q_proj=q_st,
            k_proj=k_st,
            v_proj=v_st,
            attention_dropout=dropout.st,
            out_proj=out_st,
        ),
    )
end