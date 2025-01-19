module MPIExt

using InboundsArrays

import MPI
import MPI: Buffer, UBuffer, VBuffer

@inline function Buffer(a::AbstractInboundsArray)
    return Buffer(a.a)
end

@inline function UBuffer(data::AbstractInboundsArray, count::Integer, nchunks::Union{Nothing, Integer}, datatype::MPI.Datatype)
    return UBuffer(data.a, count, nchunks, datatype)
end

@inline function VBuffer(arr::InboundsArray, counts::Vector{Cint}, displs::Vector{Cint}, dtype::MPI.Datatype)
    return VBuffer(arr.a, counts, displs, dtype)
end

end
