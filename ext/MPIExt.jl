module MPIExt

using InboundsArrays

import MPI
import MPI: Buffer, UBuffer, VBuffer

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

if !inherit_from_AbstractArray
    @inline function UBuffer(data::AbstractInboundsArray, args...)
        return UBuffer(data.a, args...)
    end
    @inline function VBuffer(data::AbstractInboundsArray, args...)
        return VBuffer(data.a, args...)
    end

    using MPI: Comm, Win
    import MPI: Scatter!, Gather!, Allgather!, Allgather, Reduce, Allreduce, Scan, Exscan,
           Neighbor_allgather!, Neighbor_allgatherv!, Win_attach!, Send

    Scatter!(sendbuf::AbstractInboundsArray{T}, recvbuf::Union{Ref{T},AbstractInboundsArray{T}}, root::Integer, comm::Comm) where {T} =
        Scatter!(UBuffer(sendbuf,length(recvbuf)), recvbuf, root, comm)
    Scatter!(sendbuf::AbstractArray{T}, recvbuf::Union{Ref{T},AbstractInboundsArray{T}}, root::Integer, comm::Comm) where {T} =
        Scatter!(UBuffer(sendbuf,length(recvbuf)), recvbuf, root, comm)
    Scatter!(sendbuf::AbstractInboundsArray{T}, recvbuf::Union{AbstractArray{T}}, root::Integer, comm::Comm) where {T} =
        Scatter!(UBuffer(sendbuf,length(recvbuf)), recvbuf, root, comm)
    Gather!(sendbuf::Union{Ref,AbstractInboundsArray}, recvbuf::AbstractInboundsArray, root::Integer, comm::Comm) =
        Gather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), root, comm)
    Gather!(sendbuf::Union{Ref,AbstractInboundsArray}, recvbuf::AbstractArray, root::Integer, comm::Comm) =
        Gather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), root, comm)
    Gather!(sendbuf::Union{Ref,AbstractArray}, recvbuf::AbstractInboundsArray, root::Integer, comm::Comm) =
        Gather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), root, comm)
    Gather(sendbuf::AbstractInboundsArray, root::Integer, comm::Comm) =
        Gather!(sendbuf, Comm_rank(comm) == root ? similar(sendbuf, Comm_size(comm) * length(sendbuf)) : nothing, root, comm)
    Allgather!(sendbuf::Union{Ref,AbstractInboundsArray}, recvbuf::AbstractInboundsArray, comm::Comm) =
        Allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), comm)
    Allgather!(sendbuf::AbstractInboundsArray, recvbuf::AbstractArray, comm::Comm) =
        Allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), comm)
    Allgather!(sendbuf::AbstractArray, recvbuf::AbstractInboundsArray, comm::Comm) =
        Allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), comm)
    Allgather(sendbuf::AbstractInboundsArray, comm::Comm) =
        Allgather!(sendbuf, similar(sendbuf, Comm_size(comm) * length(sendbuf)), comm)
    function Reduce(sendbuf::AbstractInboundsArray, op, root::Integer, comm::Comm)
        if Comm_rank(comm) == root
            Reduce!(sendbuf, similar(sendbuf), op, root, comm)
        else
            Reduce!(sendbuf, nothing, op, root, comm)
        end
    end
    Allreduce(sendbuf::AbstractInboundsArray, op, comm::Comm) =
        Allreduce!(sendbuf, similar(sendbuf), op, comm)
    Scan(sendbuf::AbstractInboundsArray, op, comm::Comm) =
        Scan!(sendbuf, similar(sendbuf), op, comm)
    Exscan(sendbuf::AbstractInboundsArray, op, comm::Comm) =
        Exscan!(sendbuf, similar(sendbuf), op, comm)
    Neighbor_allgather!(sendbuf::Union{Ref,AbstractInboundsArray}, recvbuf::AbstractInboundsArray, graph_comm::Comm) =
        Neighbor_allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), graph_comm)
    Neighbor_allgather!(sendbuf::AbstractInboundsArray, recvbuf::AbstractArray, graph_comm::Comm) =
        Neighbor_allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), graph_comm)
    Neighbor_allgather!(sendbuf::AbstractArray, recvbuf::AbstractInboundsArray, graph_comm::Comm) =
        Neighbor_allgather!(sendbuf, UBuffer(recvbuf, length(sendbuf)), graph_comm)
    Neighbor_allgatherv!(sendbuf::Union{Ref,AbstractInboundsArray}, recvbuf::AbstractInboundsArray, graph_comm::Comm) =
        Neighbor_allgatherv!(sendbuf, VBuffer(recvbuf, length(sendbuf)), graph_comm)
    Neighbor_allgatherv!(sendbuf::AbstractInboundsArray, recvbuf::AbstractArray, graph_comm::Comm) =
        Neighbor_allgatherv!(sendbuf, VBuffer(recvbuf, length(sendbuf)), graph_comm)
    Neighbor_allgatherv!(sendbuf::AbstractArray, recvbuf::AbstractInboundsArray, graph_comm::Comm) =
        Neighbor_allgatherv!(sendbuf, VBuffer(recvbuf, length(sendbuf)), graph_comm)
    function Win_attach!(win::Win, base::AbstractInboundsArray{T}) where T
        # int MPI_Win_attach(MPI_Win win, void *base, MPI_Aint size)
        API.MPI_Win_attach(win, base.a, sizeof(base))
        push!(win.object, base.a)
    end
    function Win_detach!(win::Win, base::AbstractInboundsArray{T}) where T
        # int MPI_Win_detach(MPI_Win win, const void *base)
        API.MPI_Win_detach(win, base.a)
        delete!(win.object, base.a)
    end
    Send(arr::AbstractInboundsArray, dest::Integer, tag::Integer, comm::Comm) =
        Send(Buffer(arr), dest, tag, comm)
end

end
