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

abstract type AbstractInboundsArray{T, N} <: AbstractArray{T, N} end

struct InboundsArray{T, N, TArray <: AbstractArray{T, N}} <: AbstractInboundsArray{T, N}
    a::TArray
end

InboundsVector{T, TVector} = InboundsArray{T, 1, TVector} where {T, TVector}
InboundsMatrix{T, TMatrix} = InboundsArray{T, 2, TMatrix} where {T, TMatrix}

import Base: getindex, setindex!, size, IndexStyle, length, similar, axes, BroadcastStyle,
             copyto!, copy, resize!, unsafe_convert, strides, elsize, view, maybeview,
             reshape

InboundsArray(A::InboundsArray) = A

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
    return @inbounds getindex(A.a, args...)
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

@inline function size(A::AbstractInboundsArray)
    return size(A.a)
end

@inline function IndexStyle(::InboundsArray{T, N, TArray}) where {T, N, TArray}
    return IndexStyle(TArray)
end

@inline function length(A::AbstractInboundsArray)
    return length(A.a)
end

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
BroadcastStyle(::Type{<:InboundsArray{T, N, TArray}}) where {T, N, TArray} = Broadcast.ArrayStyle{InboundsArray{T, N, TArray}}()

@inline function similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{InboundsArray{T, N, TArray}}}, ::Type{ElType}) where {T, N, TArray, ElType}
    # Scan the inputs for the InboundsArray:
    A = find_iba(bc)
    # Create the output as an InboundsArray
    similar(A, ElType, axes(bc))
end
# Special version to handle 0-d arrays, copied from Base.
@inline copy(bc::Broadcast.Broadcasted{<:Broadcast.ArrayStyle{InboundsArray{T, 0, TArray}}} where {T, TArray}) = bc[CartesianIndex()]

"`A = find_iba(As)` returns the first InboundsArray among the arguments."
find_iba(bc::Base.Broadcast.Broadcasted) = find_iba(bc.args)
find_iba(args::Tuple) = find_iba(find_iba(args[1]), Base.tail(args))
find_iba(x) = x
find_iba(::Tuple{}) = nothing
find_iba(a::InboundsArray, rest) = a
find_iba(::Any, rest) = find_iba(rest)

@inline function copyto!(A::InboundsArray, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{InboundsArray{T, N, TArray}}}) where {T, N, TArray}
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

include("LinearAlgebra_support.jl")
include("SparseArrays_support.jl")

end # module InboundsArrays
