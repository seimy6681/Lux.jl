using Pkg
ENV["GROUP"] = "CPU"  # Options are usually "CPU", "CUDA", "AMDGPU", "Metal", or "ALL"
ENV["AUTODIFF_BACKENDS"] = "Zygote,ForwardDiff"
Pkg.activate("test")
Pkg.instantiate()
using NNlib
using Test
using Lux
using LuxLib
using Random


include("../shared_testsetup.jl")

@testset "MultiHeadAttention" begin
    rng = StableRNG(12345)

    function loss(model, x, ps, st)
        y, α = first(model(x, ps, st))
        return sum(abs2, y) + sum(abs2, α)
    end

    @testset "$mode" for (mode, aType, dev, ongpu) in MODES
        dim, nheads, batch_size, len = 4, 2, 5, 3

        mha = MultiHeadAttention(dim; nheads)

        q = rand(Float32, (dim, len, batch_size)) |> aType
        k = rand(Float32, (dim, len, batch_size)) |> aType
        v = rand(Float32, (dim, len, batch_size)) |> aType

        ps, st = Lux.setup(rng, mha) |> dev

        (y, α), stₙ = mha((q, k, v), ps, st)
        @test y isa aType{Float32,3}
        @test size(y) == (dim, len, batch_size)
        @test α isa aType{Float32,4}
        @test size(α) == (len, len, nheads, batch_size)

        # check that stₙ has all the fields that st has
        @test mha((q, k, v), ps, stₙ)[1][1] ≈ y

        @testset "self-attention" begin
            (y1, α1), stₙ = mha(q, ps, st)
            (y2, α2), stₙ = mha((q, q, q), ps, stₙ)
            @test y1 ≈ y2
            @test α1 ≈ α2
        end

        @testset "key and value are the same" begin
            (y1, α1), stₙ = mha((q, k, k), ps, st)
            (y2, α2), stₙ = mha((q, k), ps, stₙ)
            @test y1 ≈ y2
            @test α1 ≈ α2
        end

        @testset "change dims" begin
            dims = 4 => 10 => 5
            nhead = 5
            mha2 = MultiHeadAttention(dims; nheads)
            ps2, st2 = Lux.setup(rng, mha2) |> dev
            (y2, _), st2ₙ = mha2((q, k, v), ps2, st2)
            @test size(y2) == (dims.second.second, len, batch_size)
        end

        @testset "mask" begin
            mask = NNlib.make_causal_mask(q)
            (y, α), stₙ = mha((q, q, q, mask), ps, st)
            @test all(α[2, 1, :, :] .== 0)
            @test α[:, :, 1, 1] ≈ triu(α[:, :, 1, 1])
        end

        @test_gradients(loss, mha, (q, k, v), ps, st; atol=1.0f-3, rtol=1.0f-3)
    end
end



rng = Random.default_rng()
Random.seed!(rng, 42)

@testset "GroupQueryAttention Implementation Tests" begin

    @testset "Constructor & Assertion Guardrails" begin
        # 1. Valid GQA construction
        gqa = GroupQueryAttention(64; nqheads=8, nkvheads=2)
        @test gqa isa AbstractLuxContainerLayer
        @test gqa.nqheads == 8
        @test gqa.nkvheads == 2

        # 2. Invalid head ratio: nqheads (8) not divisible by nkvheads (3)
        @test_throws AssertionError GroupQueryAttention(64; nqheads=8, nkvheads=3)

        # 3. Invalid dimension: qk_dim (64) not divisible by nqheads (10)
        @test_throws AssertionError GroupQueryAttention(64; nqheads=10, nkvheads=2)
    end

    @testset "Shape Verification & Input Signatures" begin
        embed_dim = 64
        q_len = 12
        kv_len = 18
        batch_size = 4
        nqheads = 8
        nkvheads = 2

        gqa = GroupQueryAttention(embed_dim; nqheads=nqheads, nkvheads=nkvheads)
        ps, st = Lux.setup(rng, gqa)

        q = randn(Float32, embed_dim, q_len, batch_size)
        k = randn(Float32, embed_dim, kv_len, batch_size)
        v = randn(Float32, embed_dim, kv_len, batch_size)

        # Single input (Self-Attention)
        (y_self, α_self), st_self = gqa(q, ps, st)
        @test size(y_self) == (embed_dim, q_len, batch_size)
        @test size(α_self) == (q_len, q_len, nqheads, batch_size)

        # 2-Tuple input (Query, Key=Value)
        (y_kv, α_kv), _ = gqa((q, k), ps, st)
        @test size(y_kv) == (embed_dim, q_len, batch_size)
        @test size(α_kv) == (kv_len, q_len, nqheads, batch_size)

        # 3-Tuple input (Query, Key, Value)
        (y_qkv, α_qkv), _ = gqa((q, k, v), ps, st)
        @test size(y_qkv) == (embed_dim, q_len, batch_size)
        @test size(α_qkv) == (kv_len, q_len, nqheads, batch_size)

        # 4-Tuple input with Causal Mask
        mask = NNlib.make_causal_mask(q)
        (y_masked, α_masked), _ = gqa((q, q, q, mask), ps, st)
        @test size(y_masked) == (embed_dim, q_len, batch_size)
        @test all(α_masked[2, 1, :, :] .== 0) # Lower triangle mask verification
    end

    @testset "Numerical Equivalence: Implicit GQA vs. Manual Expansion" begin
        embed_dim = 64
        nqheads = 8
        nkvheads = 2
        group_size = nqheads ÷ nkvheads
        seq_len = 8
        batch_size = 2

        gqa = GroupQueryAttention(embed_dim; nqheads=nqheads, nkvheads=nkvheads)
        ps, st = Lux.setup(rng, gqa)
        x = randn(Float32, embed_dim, seq_len, batch_size)

        # Forward pass through custom GQA struct
        (y_gqa, α_gqa), _ = gqa(x, ps, st)

        # Manual head expansion baseline
        q_proj, _ = gqa.q_proj(x, ps.q_proj, st.q_proj)
        k_proj, _ = gqa.k_proj(x, ps.k_proj, st.k_proj)
        v_proj, _ = gqa.v_proj(x, ps.v_proj, st.v_proj)

        d_head = embed_dim ÷ nqheads
        q_reshaped = reshape(q_proj, d_head, nqheads, seq_len, batch_size)
        k_reshaped = reshape(k_proj, d_head, nkvheads, seq_len, batch_size)
        v_reshaped = reshape(v_proj, d_head, nkvheads, seq_len, batch_size)

        # Physically duplicate Key/Value heads along axis 2
        k_repeated = repeat(k_reshaped; inner=(1, group_size, 1, 1))
        v_repeated = repeat(v_reshaped; inner=(1, group_size, 1, 1))

        x_manual, α_manual = LuxLib.scaled_dot_product_attention(
            q_reshaped, k_repeated, v_repeated; head_dim=1, token_dim=3
        )
        x_manual_flat = reshape(x_manual, embed_dim, seq_len, batch_size)
        y_manual, _ = gqa.out_proj(x_manual_flat, ps.out_proj, st.out_proj)

        @test y_gqa ≈ y_manual atol=1e-5
        @test α_gqa ≈ α_manual atol=1e-5
    end
end

@testset "$mode" for (mode, aType, dev, ongpu) in MODES

    @testset "Autodiff / Gradient Correctness" begin
        # instantiate GQA ;ayer, input/output dim of 8192, 8 query heads, 2 key/value heads
        gqa = GroupQueryAttention(64; nqheads=8, nkvheads=2)
        # initialize layer parameters (ps) and state (st) deterministically using rng
        ps, st = Lux.setup(rng, gqa) |> dev # moves ps and st from CPU to GPU if dev is a GPU device
        # generating a dummy input tensor with shape (embed_dim=8192, seq_len=8, batch_size=2)
        x = randn(Float32, 64, 8, 2) |> aType # converts x to the appropriate array type (CPU or GPU) based on aType


        # define a scalar loss function suitable for reverse-mode AD
        # 1. model(x, ps, st) return a tuple ((y, α), st_new) where y is the output tensor and α is the attention scores
        # 2. first 'first(...)' extracts '(y, α)' from the tuple returned by model(x, ps, st)
        # 3. second 'first(...)' extracts 'y' from the tuple '(y, α)'
        # 4. sum(abs2, ...) computes the sum of squared elements of L2 norm
        # loss_fn(model, x, ps, st) = sum(abs2, first(model(x, ps, st)))
        function loss_fn(model, x, ps, st)
            (y, α), _ = model(x, ps, st)
            return sum(abs2, y) + sum(abs2, α)
        end

        # compares gradients computed via Zygote/ForwardDiff against numerical finite differences between specified tolerances
        @test_gradients(loss_fn, gqa, x, ps, st; 
                        backends=[AutoZygote(), AutoForwardDiff()],
                        atol=1e-3, rtol=1e-3)

        @testset "Precision Support" begin
            gqa = GroupQueryAttention(64; nqheads=8, nkvheads=2)
            ps, st = Lux.setup(rng, gqa)
            x = randn(Float32, 64, 8, 2)

            # 1. Standard Float32 pass
            (y32, α32), _ = gqa(x, ps, st)
            @test eltype(y32) == Float32

            # 2. Float16 precision pass using Lux.f16
            ps_f16 = Lux.f16(ps)
            st_f16 = Lux.f16(st)
            x_f16  = Float16.(x)

            (y16, α16), _ = gqa(x_f16, ps_f16, st_f16)
            @test eltype(y16) == Float16
            @test eltype(α16) == Float16
        end


        using JET

        @testset "Allocation Bound Verification" begin
            st_test = Lux.testmode(st)

            # Warmup pass to trigger JIT compilation (crucial!)
            gqa(x, ps, st_test)

            # Base.@allocated measures exact bytes allocated by the expression
            allocs = @allocated gqa(x, ps, st_test)

            # Verify standard intermediate array creation stays within functional bounds
            @test allocs < 65_000
        end

        @testset "Autoregressive Shape Verification" begin
            q_single = randn(Float32, 64, 1, 2)   # Single new query token
            k_past   = randn(Float32, 64, 128, 2) # 128 cached KV tokens
            v_past   = randn(Float32, 64, 128, 2)
            
            (y_dec, α_dec), _ = gqa((q_single, k_past, v_past), ps, st)
            @test size(y_dec) == (64, 1, 2)
            @test size(α_dec) == (128, 1, 8, 2) # (kv_len, q_len, nqheads, batch)
        end

    end
end


