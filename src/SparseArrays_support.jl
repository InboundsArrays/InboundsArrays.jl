# This cannot be an Extension, because we need to define new types

export InboundsSparseMatrixCSC, InboundsSparseVector, InboundsSparseMatrixCSR

import Base: convert, copy, *, \
import LinearAlgebra: lu, lu!, mul!, Factorization
import SparseArrays: AbstractSparseMatrix, AbstractSparseMatrixCSC, SparseMatrixCSC,
                     AbstractCompressedVector, SparseVector, sparse, getcolptr, rowvals,
                     nonzeros, nonzeroinds

IbVector{T} = InboundsVector{T, Vector{T}} 

"""
Equivalent of SparseMatrixCSC, but using InboundsVector for storage
"""
struct InboundsSparseMatrixCSC{Tv, Ti <: Integer} <: AbstractSparseMatrixCSC{Tv, Ti}
    m::Int                    # Number of rows
    n::Int                    # Number of columns
    colptr::IbVector{Ti}      # Column i is in colptr[i]:(colptr[i+1]-1)
    rowval::IbVector{Ti}      # Row indices of stored values
    nzval::IbVector{Tv}       # Stored values, typically nonzeros
    parent::SparseMatrixCSC{Tv, Ti}

    function InboundsSparseMatrixCSC{Tv, Ti}(m::Integer, n::Integer, colptr::IbVector{Ti},
                                             rowval::IbVector{Ti},
                                             nzval::IbVector{Tv}) where {Tv,Ti<:Integer}
        parent = SparseMatrixCSC{Tv, Ti}(m, n, colptr.a, rowval.a, nzval.a)
        InboundsSparseMatrixCSC(parent)
    end

    function InboundsSparseMatrixCSC(parent::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
        new{Tv, Ti}(parent.m, parent.n, InboundsArray(parent.colptr),
                    InboundsArray(parent.rowval), InboundsArray(parent.nzval), parent)
    end
end

@inline function size(A::InboundsSparseMatrixCSC)
    return size(A.parent)
end

@inline function getindex(A::InboundsSparseMatrixCSC, i::Integer, j::Integer)
    return @inbounds getindex(A.parent, i, j)
end

@inline function setindex!(A::InboundsSparseMatrixCSC, v, i::Integer, j::Integer)
    return @inbounds setindex!(A.parent, v, i, j)
end

@inline function getcolptr(A::InboundsSparseMatrixCSC)
    return A.colptr
end

@inline function rowvals(A::InboundsSparseMatrixCSC)
    return A.rowval
end

@inline function nonzeros(A::InboundsSparseMatrixCSC)
    return A.nzval
end

@inline function convert(::Type{SparseMatrixCSC{Tv}}, A::InboundsSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    return A.parent
end

@inline function convert(smt::Type{SparseMatrixCSC{T}}, A::InboundsMatrix{T, TArray}) where {T, TArray}
    return InboundsSparseMatrixCSC(convert(smt, A.a))
end

@inline function convert(smt::Type{InboundsSparseMatrixCSC{T}}, A::InboundsMatrix{T, TArray}) where {T, TArray}
    return InboundsSparseMatrixCSC(convert(smt, A.a))
end

function copy(m::InboundsSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    parentcopy = copy(m.parent)
    return InboundsSparseMatrixCSC(parentcopy)
end

@inline sparse(m::InboundsSparseMatrixCSC) = copy(m)
@inline sparse(I::InboundsVector, J::InboundsVector, V::InboundsVector) = InboundsSparseMatrixCSC(sparse(I.a, J.a, V.a))

if !inherit_from_AbstractArray
    @inline sparse(m::InboundsMatrix) = InboundsSparseMatrixCSC(sparse(m.a))
    @inline function \(A::InboundsSparseMatrixCSC, x::AbstractArray)
        return InboundsArray(A.parent \ x)
    end
    @inline function \(A::InboundsSparseMatrixCSC, x::InboundsArray)
        return InboundsArray(A.parent \ x.a)
    end
end

"""
Equivalent of SparseVector, but using InboundsVector for storage
"""
struct InboundsSparseVector{Tv, Ti<:Integer} <: AbstractCompressedVector{Tv,Ti}
    n::Ti                 # Length of the sparse vector
    nzind::IbVector{Ti}   # Indices of stored values
    nzval::IbVector{Tv}   # Stored values, typically nonzeros
    parent::SparseVector{Tv,Ti}

    function InboundsSparseVector{Tv,Ti}(n::Integer, nzind::Vector{Ti}, nzval::Vector{Tv}) where {Tv,Ti<:Integer}
        parent = SparseVector(n, nzind, nzval)
        InboundsSparseVector(parent)
    end

    function InboundsSparseVector(parent::SparseVector{Tv, Ti}) where {Tv, Ti}
        new{Tv, Ti}(parent.n, InboundsArray(parent.nzind), InboundsArray(parent.nzval),
                    parent)
    end
end

@inline function size(A::InboundsSparseVector)
    return size(A.parent)
end

@inline function getindex(A::InboundsSparseVector, i::Integer)
    return @inline getindex(A.parent, i)
end

@inline function setindex!(A::InboundsSparseVector{Tv, Ti}, v::Tv, i::Ti) where {Tv, Ti <: Integer}
    return @inline setindex!(A.parent, v, i)
end

@inline function nonzeros(A::InboundsSparseVector)
    return A.nzval
end

@inline function nonzeroinds(A::InboundsSparseVector)
    return A.nzind
end

@inline function convert(::Type{SparseVector{T}}, V::InboundsSparseVector{T, TVector}) where {T, TVector}
    return V.parent
end

@inline function convert(svt::Type{SparseVector{T}}, V::InboundsVector{T, TVector}) where {T, TVector}
    return InboundsSparseVector(convert(svt, V.a))
end

@inline function convert(svt::Type{InboundsSparseVector{T}}, V::InboundsVector{T, TVector}) where {T, TVector}
    return InboundsSparseVector(convert(svt, V.a))
end

@inline function similar(A::InboundsSparseVector, type::Type{S},
                         dims::Union{Tuple{Int64}, Tuple{Int64, Int64}}) where S
    return InboundsSparseVector(similar(A.parent, type, dims))
end

function copy(v::InboundsSparseVector{Tv, Ti}) where {Tv, Ti}
    parentcopy = copy(v.parent)
    return InboundsSparseVector(parentcopy)
end

@inline sparse(v::InboundsVector) = InboundsSparseVector(sparse(v.a))
@inline sparse(v::InboundsSparseVector) = copy(v)

@inline function *(A::InboundsSparseMatrixCSC, x::InboundsSparseVector)
    return InboundsSparseVector(A.parent * x.parent)
end

@inline function *(A::InboundsSparseMatrixCSC, B::InboundsSparseMatrixCSC)
    return InboundsSparseMatrixCSC(A.parent * B.parent)
end

import SparseMatricesCSR: SparseMatrixCSR, sparsecsr

"""
Equivalent of SparseMatrixCSR, but using InboundsVector for storage
"""
struct InboundsSparseMatrixCSR{Bi, Tv, Ti <: Integer} <: AbstractSparseMatrix{Tv, Ti}
    m::Int
    n::Int
    rowptr::IbVector{Ti}
    colval::IbVector{Ti}
    nzval::IbVector{Tv}
    parent::SparseMatrixCSR{Bi, Tv, Ti}

    function InboundsSparseMatrixCSR{Bi, Tv, Ti}(m::Integer, n::Integer,
                                                 rowptr::IbVector{Ti},
                                                 colval::IbVector{Ti},
                                                 nzval::IbVector{Tv}) where {Bi, Tv, Ti<:Integer}
        parent = SparseMatrixCSR{Bi, Tv, Ti}(m, n, rowptr.a, colval.a, nzval.a)
        InboundsSparseMatrixCSR(parent)
    end

    function InboundsSparseMatrixCSR(parent::SparseMatrixCSR{Bi, Tv, Ti}) where {Bi, Tv, Ti}
        new{Bi, Tv, Ti}(parent.m, parent.n, InboundsArray(parent.rowptr),
                        InboundsArray(parent.colval), InboundsArray(parent.nzval), parent)
    end
end

@inline function size(A::InboundsSparseMatrixCSR)
    return size(A.parent)
end

@inline function getindex(A::InboundsSparseMatrixCSR, i::Integer, j::Integer)
    return @inline getindex(A.parent, i, j)
end

@inline function setindex!(A::InboundsSparseMatrixCSR, v, i::Integer, j::Integer)
    return @inline setindex!(A.parent, v, i, j)
end

@inline function convert(::Type{SparseMatrixCSR}, A::InboundsSparseMatrixCSR)
    return A.parent
end

@inline function convert(t::Type{SparseMatrixCSR{Bi, T, Ti}}, A::InboundsMatrix{T, TMatrix}) where {Bi, T, Ti, TMatrix}
    return InboundsSparseMatrixCSR(convert(t, A.a))
end

@inline function convert(t::Type{InboundsSparseMatrixCSR{Bi, T, Ti}}, A::InboundsMatrix{T, TMatrix}) where {Bi, T, Ti, TMatrix}
    return InboundsSparseMatrixCSR(convert(SparseMatrixCSR{Bi, T, Ti}, A.a))
end

function copy(m::InboundsSparseMatrixCSR{Bi, Tv, Ti}) where {Bi, Tv, Ti}
    parentcopy = copy(m.parent)
    return InboundsSparseMatrixCSR{Bi}(parentcopy)
end

@inline function sparsecsr(I::InboundsVector, J::InboundsVector, V::InboundsVector, args...)
    return InboundsSparseMatrixCSR(sparsecsr(I.a, J.a, V.a, args...))
end

@inline function sparsecsr(bi::Val{Bi}, I::InboundsVector, J::InboundsVector, V::InboundsVector, args...) where Bi
    return InboundsSparseMatrixCSR(sparsecsr(bi, I.a, J.a, V.a, args...))
end

@inline sparsecsr(m::InboundsSparseMatrixCSR) = copy(m)

# Ensure some LinearAlgebra functions can work correctly - they do not need InboundsArrays
# as we assume they already use `@inbounds` where appropriate.
@inline function lu(m::InboundsSparseMatrixCSC{Tv, Ti} where {Tv, Ti})
    return lu(m.parent)
end
@inline function lu!(mlu::Factorization, m::InboundsSparseMatrixCSC{Tv, Ti} where {Tv, Ti})
    return lu!(mlu, m.parent)
end
# Need to use specific enough types for `mul!()` arguments that these versions take
# precedence over AbstractVector versions defined in LinearAlgebra.
@inline function mul!(C::InboundsMatrix, A::InboundsSparseMatrixCSC, B::InboundsMatrix, α::Number, β::Number)
    mul!(C.a, A.parent, B.a, α, β)
    return C
end
@inline function mul!(C::InboundsSparseMatrixCSC, A::InboundsSparseMatrixCSC, B::InboundsMatrix, α::Number, β::Number)
    mul!(C.parent, A.parent, B.a, α, β)
    return C
end
@inline function mul!(C::InboundsMatrix, A::InboundsSparseMatrixCSC, B::InboundsSparseMatrixCSC, α::Number, β::Number)
    mul!(C.a, A.parent, B.parent, α, β)
    return C
end
@inline function mul!(C::InboundsSparseMatrixCSC, A::InboundsSparseMatrixCSC, B::InboundsSparseMatrixCSC, α::Number, β::Number)
    mul!(C.parent, A.parent, B.parent, α, β)
    return C
end
@inline function mul!(C::InboundsVector, A::InboundsSparseMatrixCSC, B::InboundsVector, α::Number, β::Number)
    mul!(C.a, A.parent, B.a, α, β)
    return C
end
@inline function mul!(C::InboundsSparseVector, A::InboundsSparseMatrixCSC, B::InboundsVector, α::Number, β::Number)
    mul!(C.parent, A.parent, B.a, α, β)
    return C
end
@inline function mul!(C::InboundsVector, A::InboundsSparseMatrixCSC, B::InboundsSparseVector, α::Number, β::Number)
    mul!(C.a, A.parent, B.parent, α, β)
    return C
end
@inline function mul!(C::InboundsSparseVector, A::InboundsSparseMatrixCSC, B::InboundsSparseVector, α::Number, β::Number)
    mul!(C.parent, A.parent, B.parent, α, β)
    return C
end
@inline function lu(m::InboundsSparseMatrixCSR{Bi, Tv, Ti} where {Bi, Tv, Ti})
    return lu(m.parent)
end
@inline function lu!(mlu::Factorization, m::InboundsSparseMatrixCSR{Bi, Tv, Ti} where {Bi, Tv, Ti})
    return lu!(m.parent)
end
# Need to use specific enough types for `mul!()` arguments that these versions take
# precedence over AbstractVector versions defined in LinearAlgebra.
@inline function mul!(C::InboundsMatrix, A::InboundsSparseMatrixCSR, B::InboundsMatrix, α::Number, β::Number)
    mul!(C.a, A.parent, B.a, α, β)
    return C
end
@inline function mul!(C::InboundsVector, A::InboundsSparseMatrixCSR, B::InboundsVector, α::Number, β::Number)
    mul!(C.a, A.parent, B.a, α, β)
    return C
end

@inline function *(A::InboundsSparseMatrixCSR, x::InboundsSparseVector)
    return InboundsSparseVector(A.parent * x.parent)
end
