"""
InboundsArrays gives a convenient way to remove bounds-checks on array accesses, without
needing to using `@inbounds` in many places or use `--check-bounds=no`. The effect is to
apply `@inbounds` to all `getindex()` and `setindex!()` calls.

An InboundsArray can wrap any other array type. For example for `Array`/`Vector`/`Matrix`:
```julia
a = InboundsArray(rand(3,4,5))
b = similar(a)
for i ∈ 1:5, j ∈ 1:4, k ∈ 1:3
    b[k,j,i] = a[k,j,i] + 1
end
c = a .* 2

v = InboundsVector(rand(6))
M = InboundsMatrix(rand(7,8))
```
Broadcasting is implemented so that the result of broadcasting an InboundsArray with any
other array types is an InboundsArray (if the result is an array and not a scalar). So `c`
in the example above is an `InboundsArray`.


Testing
-------

As bounds checks (on array accesses) are disabled by default when using `InboundsArray`,
you should make sure to test your package using `--check-bounds=yes`, which will restore
the bounds checks.
"""
module InboundsArrays

export InboundsArray, InboundsVector, InboundsMatrix, AbstractInboundsArray,
       InboundsSparseMatrixCSC, InboundsSparseVector, InboundsSparseMatrixCSR,
       get_noninbounds

using Preferences

const default_inherit_from_AbstractArray = false
const inherit_from_AbstractArray = @load_preference("inherit_from_AbstractArray", default_inherit_from_AbstractArray)

"""
    set_inherit_from_AbstractArray(value::Bool=$default_inherit_from_AbstractArray)

Set `inherit_from_AbstractArray`. By default, when this is set to `false`, `InboundsArray`
does not inherit from `AbstractArray`. This should be safer, as situations that would
result in degraded performance due to using an `AbstractArray` implementation rather than
a more specific, optimized one (e.g. for `Array`) should error rather than running slowly.
When this happens, a wrapper function should be added to `InboundsArrays.jl` to provide an
`InboundsArray` interface for the function in question, that passes in the wrapped array
rather than the `InboundsArray`. To work without errors, but with the risk of slow
execution in unsupported cases, call
```julia
InboundsArrays.set_inherit_from_AbstractArray(true)
```
This setting will be saved in `LocalPreferences.toml` and so will persist. Call again with
no argument (or `false`) to reset to the default.

After calling this function, restart Julia (and recompile any system images that include
`InboundsArrays`) in order for the setting to take effect.
"""
function set_inherit_from_AbstractArray(value::Bool=default_inherit_from_AbstractArray)
    @set_preferences!("inherit_from_AbstractArray" => value)
    @info "`inherit_from_AbstractArray = $value` is set. Restart Julia for it to take effect."
end

if inherit_from_AbstractArray
    abstract type AbstractInboundsArray{T, N} <: AbstractArray{T, N} end
else
    abstract type AbstractInboundsArray{T, N} end
end

"""
    InboundsArray{T, N, TArray <: AbstractArray{T, N}} <: AbstractInboundsArray{T, N}

Wrapper array type that disables bounds checks for array accesses, but otherwise forwards
all calls to the wrapped `TArray` array.

Bounds checks can be enabled for testing by running Julia with `--check-bounds=yes`.

Construct with one of:
* `InboundsArray(A::AbstractArray)`
    Creates an `InboundsArray` that wraps `A`.
* `InboundsArray{T}(initializer, dims...)`
    Creates an `InboundsArray` wrapping an `Array{T}(initializer, dims...)`.
* `InboundsArray{T, N, TArray}(initializer, dims...)`
    Creates an `InboundsArray` wrapping a `TArray(initializer, dims)` where `TArray <: AbstractArray{T, N}`.
"""
struct InboundsArray{T, N, TArray <: AbstractArray{T, N}} <: AbstractInboundsArray{T, N}
    a::TArray
end

"""
    InboundsVector{T, TVector <: AbstractVector{T}} <: AbstractInboundsArray{T, 1}

Wrapper vector type that disables bounds checks for array accesses, but otherwise forwards
all calls to the wrapped `TVector` array.

Bounds checks can be enabled for testing by running Julia with `--check-bounds=yes`.

Construct with one of:
* `InboundsVector(V::AbstractVector)`
    Creates an `InboundsVector` that wraps `V`.
* `InboundsVector{T}(initializer, n)`
    Creates an `InboundsVector` wrapping a `Vector{T}(initializer, n)`.
* `InboundsVector{T, TVector}(initializer, n)`
    Creates an `InboundsVector` wrapping a `TVector(initializer, n)` where `TVector <: AbstractVector{T}`.
"""
InboundsVector{T, TVector} = InboundsArray{T, 1, TVector} where {T, TVector}

"""
    InboundsMatrix{T, TMatrix <: AbstractMatrix{T}} <: AbstractInboundsArray{T, 2}

Wrapper matrix type that disables bounds checks for array accesses, but otherwise forwards
all calls to the wrapped `TMatrix` array.

Bounds checks can be enabled for testing by running Julia with `--check-bounds=yes`.

Construct with one of:
* `InboundsMatrix(M::AbstractMatrix)`
    Creates an `InboundsMatrix` that wraps `M`.
* `InboundsMatrix{T}(initializer, m, n)`
    Creates an `InboundsMatrix` wrapping a `Matrix{T}(initializer, m, n)`.
* `InboundsMatrix{T, TMatrix}(initializer, n)`
    Creates an `InboundsMatrix` wrapping a `TMatrix(initializer, m, n)` where `TMatrix <: AbstractMatrix{T}`.
"""
InboundsMatrix{T, TMatrix} = InboundsArray{T, 2, TMatrix} where {T, TMatrix}

import Base: getindex, setindex!, size, IndexStyle, length, ndims, similar, axes,
             BroadcastStyle, copyto!, copy, resize!, unsafe_convert, strides, elsize,
             view, maybeview, reshape, selectdim, isapprox, iterate, eachindex,
             broadcastable, vec, *, adjoint, transpose, inv, lastindex, isassigned,
             reverse!, reverse, push!, pop!, sum, prod, maximum, minimum, all, any,
             extrema, searchsorted, searchsortedfirst, searchsortedlast, findfirst,
             findlast, findnext, findprev, findall, findmax, findmin, findmax!, findmin!

@inline InboundsArray(A::InboundsArray) = A

# This version handles any scalar `A`, because the default constructor for the `struct`
# defines a more-specific method for any `A` that is an `AbstractArray`
@inline InboundsArray(A) = A

function InboundsArray{T}(arg1, arg2, args...) where T
    return InboundsArray(Array{T}(arg1, arg2, args...))
end

function InboundsArray{T, N, TArray}(arg1, arg2, args...) where {T, N, TArray}
    return InboundsArray(TArray(arg1, arg2, args...))
end

function InboundsVector(v::AbstractVector{T}) where T
    return InboundsArray(v)
end

function InboundsVector{T}(arg1, arg2) where T
    return InboundsVector(Vector{T}(arg1, arg2))
end

function InboundsVector{T, TVector}(arg1, arg2) where {T, TVector}
    return InboundsVector(TVector(arg1, arg2))
end

function InboundsMatrix(m::AbstractMatrix{T}) where T
    return InboundsArray(m)
end

function InboundsMatrix{T}(arg1, arg2, arg3) where T
    return InboundsMatrix(Matrix{T}(arg1, arg2, arg3))
end

function InboundsMatrix{T, TMatrix}(arg1, arg2, arg3) where {T, TMatrix}
    return InboundsMatrix(TMatrix(arg1, arg2, arg3))
end

"""
    get_noninbounds(A)

If `A` is an `InboundsArray{T, N, TArray}`, returns the wrapped array of type `TArray`,
otherwise returns `A` unchanged.
"""
function get_noninbounds end
@inline get_noninbounds(A::AbstractInboundsArray) = A.a
@inline get_noninbounds(A) = A

@inline function getindex(A::AbstractInboundsArray, args...)
    return @inbounds InboundsArray(getindex(A.a, args...))
end

@inline function setindex!(A::AbstractInboundsArray, v, i::Int)
    return @inbounds setindex!(A.a, v, i)
end

@inline function setindex!(A::AbstractInboundsArray{T, N}, v, I::Vararg{Int, N}) where {T, N}
    return @inbounds setindex!(A.a, v, I...)
end

@inline function setindex!(A::AbstractInboundsArray, X, I...)
    return @inbounds setindex!(A.a, X, I...)
end

@inline function size(A::AbstractInboundsArray, args...)
    return size(A.a, args...)
end

@inline function IndexStyle(::InboundsArray{T, N, TArray}) where {T, N, TArray}
    return IndexStyle(TArray)
end
@inline IndexStyle(A::AbstractInboundsArray, B::AbstractInboundsArray) = IndexStyle(IndexStyle(A), IndexStyle(B))
@inline IndexStyle(A::AbstractInboundsArray, B::AbstractArray) = IndexStyle(IndexStyle(A), IndexStyle(B))
@inline IndexStyle(A::AbstractArray, B::AbstractInboundsArray) = IndexStyle(IndexStyle(A), IndexStyle(B))
@inline IndexStyle(A::AbstractInboundsArray, B...) = IndexStyle(IndexStyle(A), IndexStyle(B...))
@inline IndexStyle(A::AbstractArray, B::AbstractInboundsArray, C...) = IndexStyle(IndexStyle(A), IndexStyle(B, IndexStyle(C...)))

@inline function length(A::AbstractInboundsArray)
    return length(A.a)
end

@inline ndims(A::AbstractInboundsArray{T, N}) where {T, N} = N

@inline function similar(A::InboundsArray)
    return InboundsArray(similar(A.a))
end

@inline function similar(A::InboundsArray, type::Type{S}) where S
    return InboundsArray(similar(A.a, type))
end

@inline function similar(A::InboundsArray, dims::Dims)
    return InboundsArray(similar(A.a, dims))
end

@inline function similar(A::InboundsArray, type::Type{S}, dims::Dims) where S
    return InboundsArray(similar(A.a, type, dims))
end

@inline function similar(A::InboundsArray, type::Type{S}, dims::Tuple{Int64, Vararg{Int64, N}}) where {S, N}
    return InboundsArray(similar(A.a, type, dims))
end

@inline function axes(A::AbstractInboundsArray)
    return axes(A.a)
end

# Define these so that a broadcast operations with an InboundsArray return an
# InboundsArray - see https://docs.julialang.org/en/v1/manual/interfaces/#Selecting-an-appropriate-output-array
struct InboundsArrayStyle{T, N} <: Broadcast.AbstractArrayStyle{Any} end
BroadcastStyle(::Type{<:AbstractInboundsArray{T, N}}) where {T, N} = InboundsArrayStyle{T, N}()

@inline function similar(bc::Broadcast.Broadcasted{InboundsArrayStyle{T, N}}, ::Type{ElType}) where {T, N, ElType}
    # Scan the inputs for the InboundsArray:
    A = find_iba(bc)
    # Create the output as an InboundsArray
    similar(A, ElType, axes(bc))
end
# Special version to handle 0-d arrays, copied from Base.
@inline copy(bc::Broadcast.Broadcasted{<:InboundsArrayStyle{InboundsArray{T, 0, TArray}}} where {T, TArray}) = bc[CartesianIndex()]

"`A = find_iba(As)` returns the first InboundsArray among the arguments."
find_iba(bc::Base.Broadcast.Broadcasted) = find_iba(bc.args)
find_iba(args::Tuple) = find_iba(find_iba(args[1]), Base.tail(args))
find_iba(x) = x
find_iba(::Tuple{}) = nothing
find_iba(a::InboundsArray, rest) = a
find_iba(::Any, rest) = find_iba(rest)

@inline function copyto!(A::InboundsArray, bc::Broadcast.Broadcasted{InboundsArrayStyle{InboundsArray{T, N, TArray}}}) where {T, N, TArray}
    @inbounds copyto!(A.a, bc)
    return A
end

@inline function resize!(a::InboundsVector, nl::Integer)
    resize!(a.a, nl)
    return a
end

@inline function reshape(a::AbstractInboundsArray, dims::Int64...)
    return InboundsArray(reshape(a.a, dims...))
end

@inline function selectdim(a::AbstractInboundsArray, d::Integer, i)
    return InboundsArray(selectdim(a.a, d, i))
end

@inline function unsafe_convert(pt::Type{Ptr{T}}, a::InboundsArrays.InboundsArray{T, N, TArray}) where {T, N, TArray}
    return unsafe_convert(pt, a.a)
end

@inline function strides(a::AbstractInboundsArray)
    return strides(a.a)
end

@inline function elsize(a::AbstractInboundsArray)
    return elsize(a.a)
end

@inline function view(a::AbstractInboundsArray, I::Vararg{Any,M}) where M
    return InboundsArray(view(a.a, I...))
end

@inline function maybeview(a::AbstractInboundsArray, I::Vararg{Any,M}) where M
    return InboundsArray(maybeview(a.a, I...))
end

@inline function *(A::InboundsArray{TA, NA, TArrayA}, x::AbstractArray{Tx, Nx}) where {TA, NA, TArrayA, Tx, Nx}
    return InboundsArray(A.a * x)
end
@inline function *(A::AbstractMatrix{TA}, x::InboundsArray{Tx, Nx, TArrayx}) where {TA, Tx, Nx, TArrayx}
    return InboundsArray(A * x.a)
end
@inline function *(A::InboundsArray{Tm, 2, TArrayA}, x::InboundsArray{Tx, Nx, TArrayx}) where {Tm, TArrayA, Tx, Nx, TArrayx}
    return InboundsArray(A.a * x.a)
end

@inline function reverse!(v::AbstractInboundsArray{T, 1} where T)
    reverse!(v.a)
    return v
end

@inline function reverse(v::InboundsVector)
    return InboundsVector(reverse(v.a))
end

@inline function push!(v::AbstractInboundsArray{T, 1} where T, args...)
    push!(v.a, args...)
    return v
end

@inline function pop!(v::AbstractInboundsArray{T, 1} where T, args...)
    return pop!(v.a, args...)
end

@inline adjoint(A::InboundsArray) = InboundsArray(adjoint(A.a))
@inline transpose(A::InboundsArray) = InboundsArray(transpose(A.a))
@inline inv(A::InboundsArray) = InboundsArray(inv(A.a))

if !inherit_from_AbstractArray
    @inline function copy(A::InboundsArray)
        return InboundsArray(@inbounds copy(A.a))
    end
    @inline function copyto!(A::AbstractInboundsArray, B::AbstractArray)
        return @inbounds copyto!(A.a, B)
    end
    @inline function copyto!(A::AbstractInboundsArray, bc::Broadcast.Broadcasted)
        @inbounds copyto!(A.a, bc)
        return A
    end
    @inline vec(A::AbstractInboundsArray) = reshape(A, length(A))
    @inline lastindex(A::AbstractInboundsArray, args...) = lastindex(A.a, args...)
    @inline isassigned(A::AbstractInboundsArray, args...) = isassigned(A.a, args...)

    # Copied from the AbstractArray implementation in base/abstractaray.jl
    similar(a::AbstractInboundsArray{T}) where {T} = similar(a, T)
    similar(a::AbstractInboundsArray, ::Type{T}) where {T} = similar(a, T, Base.to_shape(axes(a)))
    similar(a::AbstractInboundsArray{T}, dims::Tuple) where {T} = similar(a, T, Base.to_shape(dims))
    similar(a::AbstractInboundsArray{T}, dims::Base.DimOrInd...) where {T} = similar(a, T, Base.to_shape(dims))
    similar(a::AbstractInboundsArray, ::Type{T}, dims::Base.DimOrInd...) where {T} = similar(a, T, Base.to_shape(dims))
    similar(a::AbstractInboundsArray, ::Type{T}, dims::Tuple{Union{Integer, Base.OneTo}, Vararg{Union{Integer, Base.OneTo}}}) where {T} = similar(a, T, Base.to_shape(dims))
    function iterate(A::AbstractInboundsArray, state=(eachindex(A),))
        y = iterate(state...)
        y === nothing && return nothing
        A[y[1]], (state[1], Base.tail(y)...)
    end
    axes1(A::AbstractInboundsArray{<:Any,0}) = Base.OneTo(1)
    axes1(A::AbstractInboundsArray) = (@inline; axes(A)[1])
    axes1(iter) = Base.oneto(length(iter))
    eachindex(A::AbstractInboundsArray) = (@inline(); eachindex(IndexStyle(A), A))
    function eachindex(A::AbstractInboundsArray, B::AbstractInboundsArray)
        @inline
        eachindex(IndexStyle(A,B), A, B)
    end
    function eachindex(A::AbstractInboundsArray, B::AbstractInboundsArray...)
        @inline
        eachindex(IndexStyle(A,B...), A, B...)
    end
    eachindex(::IndexLinear, A::AbstractInboundsArray) = (@inline; Base.oneto(length(A)))
    eachindex(::IndexLinear, A::AbstractInboundsArray{T, 1} where {T}) = (@inline; axes1(A))
    function eachindex(::IndexLinear, A::AbstractInboundsArray, B::AbstractInboundsArray...)
        @inline
        indsA = eachindex(IndexLinear(), A)
        Base._all_match_first(X->eachindex(IndexLinear(), X), indsA, B...) ||
            throw_eachindex_mismatch_indices(IndexLinear(), eachindex(A), eachindex.(B)...)
        indsA
    end
    broadcastable(x::AbstractInboundsArray) = x

    @inline function isapprox(x::AbstractInboundsArray, y; kwargs...)
        return isapprox(x.a, y; kwargs...)
    end
    @inline function isapprox(x, y::AbstractInboundsArray; kwargs...)
        return isapprox(x, y.a; kwargs...)
    end
    @inline function isapprox(x::AbstractInboundsArray, y::AbstractInboundsArray; kwargs...)
        return isapprox(x.a, y.a; kwargs...)
    end

    @inline sum(A::AbstractInboundsArray, args...; kwargs...) = sum(A.a, args...; kwargs...)
    @inline prod(A::AbstractInboundsArray, args...; kwargs...) = prod(A.a, args...; kwargs...)
    @inline maximum(A::AbstractInboundsArray, args...; kwargs...) = maximum(A.a, args...; kwargs...)
    @inline minimum(A::AbstractInboundsArray, args...; kwargs...) = minimum(A.a, args...; kwargs...)
    @inline extrema(A::AbstractInboundsArray, args...; kwargs...) = extrema(A.a, args...; kwargs...)
    @inline all(A::AbstractInboundsArray, args...; kwargs...) = all(A.a, args...; kwargs...)
    @inline any(A::AbstractInboundsArray, args...; kwargs...) = any(A.a, args...; kwargs...)
    @inline searchsorted(v::AbstractInboundsArray, x; kwargs...) = searchsorted(v.a, x; kwargs...)
    @inline searchsortedfirst(A::AbstractInboundsArray, args...; kwargs...) = searchsortedfirst(A.a, args...; kwargs...)
    @inline searchsortedlast(A::AbstractInboundsArray, args...; kwargs...) = searchsortedlast(A.a, args...; kwargs...)
    @inline findfirst(A::AbstractInboundsArray) = findfirst(A.a)
    @inline findfirst(p::Function, A::AbstractInboundsArray, args...; kwargs...) = findfirst(p, A.a, args...; kwargs...)
    @inline findfirst(p::AbstractInboundsArray, A::AbstractInboundsArray, args...; kwargs...) = findfirst(p.a, A.a, args...; kwargs...)
    @inline findfirst(p, A::AbstractInboundsArray, args...; kwargs...) = findfirst(p, A.a, args...; kwargs...)
    @inline findlast(A::AbstractInboundsArray) = findlast(A.a)
    @inline findlast(p::Function, A::AbstractInboundsArray, args...; kwargs...) = findlast(p, A.a, args...; kwargs...)
    @inline findlast(p::AbstractInboundsArray, A::AbstractInboundsArray, args...; kwargs...) = findlast(p.a, A.a, args...; kwargs...)
    @inline findlast(p, A::AbstractInboundsArray, args...; kwargs...) = findlast(p, A.a, args...; kwargs...)
    @inline findnext(A::AbstractInboundsArray, i) = findnext(A.a, i)
    @inline findnext(p::Function, A::AbstractInboundsArray, args...; kwargs...) = findnext(p, A.a, args...; kwargs...)
    @inline findnext(p::AbstractInboundsArray, A::AbstractInboundsArray, args...; kwargs...) = findnext(p.a, A.a, args...; kwargs...)
    @inline findnext(p, A::AbstractInboundsArray, args...; kwargs...) = findnext(p, A.a, args...; kwargs...)
    @inline findprev(A::AbstractInboundsArray, i) = findprev(A.a, i)
    @inline findprev(p::Function, A::AbstractInboundsArray, args...; kwargs...) = findprev(p, A.a, args...; kwargs...)
    @inline findprev(p::AbstractInboundsArray, A::AbstractInboundsArray, args...; kwargs...) = findprev(p.a, A.a, args...; kwargs...)
    @inline findprev(p, A::AbstractInboundsArray, args...; kwargs...) = findprev(p, A.a, args...; kwargs...)
    @inline findall(A::AbstractInboundsArray) = findall(A.a)
    @inline findall(p, A::AbstractInboundsArray, args...; kwargs...) = findall(p, A.a, args...; kwargs...)
    @inline findmax(A::AbstractInboundsArray, args...; kwargs...) = findmax(A.a, args...; kwargs...)
    @inline findmin(A::AbstractInboundsArray, args...; kwargs...) = findmin(A.a, args...; kwargs...)
    @inline findmax!(rval, rind, A::AbstractInboundsArray, args...; kwargs...) = findmax!(rval, rind, A.a, args...; kwargs...)
    @inline findmin!(rval, rind, A::AbstractInboundsArray, args...; kwargs...) = findmin!(rval, rind, A.a, args...; kwargs...)
end

include("LinearAlgebra_support.jl")
include("SparseArrays_support.jl")

end # module InboundsArrays
