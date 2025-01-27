module NCDatasetsExt

using InboundsArrays

import NCDatasets
import Base: setindex!

@inline function setindex!(dset::NCDatasets.Variable, x::AbstractInboundsArray, I...)
    return @inbounds setindex!(dset, x.a, I...)
end

end # module NCDatasetsExt
