module HDF5Ext

using InboundsArrays

import HDF5
import Base: setindex!

@inline function setindex!(dset::HDF5.Dataset, x::AbstractInboundsArray, I::HDF5.IndexType...)
    return @inbounds setindex!(dset, x.a, I...)
end

@inline function setindex!(dset::HDF5.File, x::AbstractInboundsArray, path::String)
    return @inbounds setindex!(dset, x.a, path)
end

end # module HDF5Ext
