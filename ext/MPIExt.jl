module MPIExt

using InboundsArrays

import MPI
using MPI: Comm, Comm_rank, Comm_size, Win
import MPI: Buffer, UBuffer, VBuffer,  Scatter!, Gather!, Gather, Allgather!, Allgather,
            Reduce, Allreduce, Scan, Exscan, Neighbor_allgather!, Neighbor_allgatherv!,
            Win_attach!, Win_detach!, Send

const inherit_from_AbstractArray = InboundsArrays.inherit_from_AbstractArray

@inline function Buffer(a::AbstractInboundsArray)
    return Buffer(a.a)
end

@inline function UBuffer(data::AbstractInboundsArray, count::Integer, nchunks::Union{Nothing, Integer}, datatype::MPI.Datatype)
    return UBuffer(data.a, count, nchunks, datatype)
end

@inline function VBuffer(arr::InboundsArray, counts::Vector{Cint}, displs::Vector{Cint}, dtype::MPI.Datatype)
    return VBuffer(arr.a, counts, displs, dtype)
end

@inline function UBuffer(data::AbstractInboundsArray, count::Integer)
    return UBuffer(data.a, count)
end

@inline function VBuffer(arr::InboundsArray, args...)
    return VBuffer(arr.a, args...)
end
@inline function VBuffer(arr::InboundsArray, counts::InboundsArray, args...)
    return VBuffer(arr.a, counts.a, args...)
end
@inline function VBuffer(arr::InboundsArray, counts::InboundsArray, displs::InboundsArray, args...)
    return VBuffer(arr.a, counts.a, displs.a, args...)
end


Scatter!(sendbuf::AbstractInboundsArray{T}, recvbuf::AbstractInboundsArray{T}, root::Integer, comm::Comm) where {T} =
    InboundsArray(Scatter!(UBuffer(sendbuf,length(recvbuf)), recvbuf, root, comm))
Scatter!(sendbuf::AbstractArray{T}, recvbuf::AbstractInboundsArray{T}, root::Integer, comm::Comm) where {T} =
    InboundsArray(Scatter!(UBuffer(sendbuf,length(recvbuf)), recvbuf, root, comm))
Scatter!(sendbuf::AbstractInboundsArray{T}, recvbuf::AbstractArray{T}, root::Integer, comm::Comm) where {T} =
    InboundsArray(Scatter!(UBuffer(sendbuf,length(recvbuf)), recvbuf, root, comm))
Gather!(sendbuf::AbstractInboundsArray, recvbuf::AbstractInboundsArray, root::Integer, comm::Comm) =
    InboundsArray(Gather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), root, comm))
Gather!(sendbuf::Ref, recvbuf::AbstractInboundsArray, root::Integer, comm::Comm) =
    InboundsArray(Gather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), root, comm))
Gather!(sendbuf::AbstractInboundsArray, recvbuf::AbstractArray, root::Integer, comm::Comm) =
    InboundsArray(Gather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), root, comm))
Gather!(sendbuf::AbstractArray, recvbuf::AbstractInboundsArray, root::Integer, comm::Comm) =
    InboundsArray(Gather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), root, comm))
# The call to similar() should ensure that this Gather() returns an InboundsArray
Gather(sendbuf::AbstractInboundsArray, root::Integer, comm::Comm) =
    Gather!(sendbuf, Comm_rank(comm) == root ? similar(sendbuf, Comm_size(comm) * length(sendbuf)) : nothing, root, comm)
Gather(sendbuf::AbstractInboundsArray, comm::Comm; root::Integer=Cint(0)) =
    Gather(sendbuf, root, comm)
Allgather!(sendbuf::AbstractInboundsArray, recvbuf::AbstractInboundsArray, comm::Comm) =
    InboundsArray(Allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), comm))
Allgather!(sendbuf::Ref, recvbuf::AbstractInboundsArray, comm::Comm) =
    InboundsArray(Allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), comm))
Allgather!(sendbuf::AbstractInboundsArray, recvbuf::AbstractArray, comm::Comm) =
    InboundsArray(Allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), comm))
Allgather!(sendbuf::AbstractArray, recvbuf::AbstractInboundsArray, comm::Comm) =
    InboundsArray(Allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), comm))
Allgather(sendbuf::AbstractInboundsArray, comm::Comm) =
    InboundsArray(Allgather!(sendbuf, similar(sendbuf, Comm_size(comm) * length(sendbuf)), comm))
function Reduce(sendbuf::AbstractInboundsArray, op, root::Integer, comm::Comm)
    if Comm_rank(comm) == root
        InboundsArray(Reduce!(sendbuf, similar(sendbuf), op, root, comm))
    else
        InboundsArray(Reduce!(sendbuf, nothing, op, root, comm))
    end
end
Allreduce(sendbuf::AbstractInboundsArray, op, comm::Comm) =
    InboundsArray(Allreduce!(sendbuf, similar(sendbuf), op, comm))
Scan(sendbuf::AbstractInboundsArray, op, comm::Comm) =
    InboundsArray(Scan!(sendbuf, similar(sendbuf), op, comm))
Exscan(sendbuf::AbstractInboundsArray, op, comm::Comm) =
    InboundsArray(Exscan!(sendbuf, similar(sendbuf), op, comm))
Neighbor_allgather!(sendbuf::AbstractInboundsArray, recvbuf::AbstractInboundsArray, graph_comm::Comm) =
    InboundsArray(Neighbor_allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), graph_comm))
Neighbor_allgather!(sendbuf::Ref, recvbuf::AbstractInboundsArray, graph_comm::Comm) =
    InboundsArray(Neighbor_allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), graph_comm))
Neighbor_allgather!(sendbuf::AbstractInboundsArray, recvbuf::AbstractArray, graph_comm::Comm) =
    InboundsArray(Neighbor_allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), graph_comm))
Neighbor_allgather!(sendbuf::AbstractArray, recvbuf::AbstractInboundsArray, graph_comm::Comm) =
    InboundsArray(Neighbor_allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), graph_comm))
Neighbor_allgatherv!(sendbuf::AbstractInboundsArray, recvbuf::AbstractInboundsArray, graph_comm::Comm) =
    InboundsArray(Neighbor_allgatherv!(sendbuf, VBuffer(recvbuf, length(sendbuf)), graph_comm))
Neighbor_allgatherv!(sendbuf::Ref, recvbuf::AbstractInboundsArray, graph_comm::Comm) =
    InboundsArray(Neighbor_allgatherv!(sendbuf, VBuffer(recvbuf, length(sendbuf)), graph_comm))
Neighbor_allgatherv!(sendbuf::AbstractInboundsArray, recvbuf::AbstractArray, graph_comm::Comm) =
    InboundsArray(Neighbor_allgatherv!(sendbuf, VBuffer(recvbuf, length(sendbuf)), graph_comm))
Neighbor_allgatherv!(sendbuf::AbstractArray, recvbuf::AbstractInboundsArray, graph_comm::Comm) =
    InboundsArray(Neighbor_allgatherv!(sendbuf, VBuffer(recvbuf, length(sendbuf)), graph_comm))
function Win_attach!(win::Win, base::AbstractInboundsArray{T}) where T
    # int MPI_Win_attach(MPI_Win win, void *base, MPI_Aint size)
    MPI.API.MPI_Win_attach(win, base.a, sizeof(base))
    push!(win.object, base.a)
end
function Win_detach!(win::Win, base::AbstractInboundsArray{T}) where T
    # int MPI_Win_detach(MPI_Win win, const void *base)
    MPI.API.MPI_Win_detach(win, base.a)
    delete!(win.object, base.a)
end
Send(arr::AbstractInboundsArray, dest::Integer, tag::Integer, comm::Comm) =
    Send(Buffer(arr), dest, tag, comm)

end
