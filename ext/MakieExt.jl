module MakieExt

using InboundsArrays

import Makie: convert_single_argument

# Tell Makie how to convert an InboundsArray to the wrapped array type.
@inline convert_single_argument(x::AbstractInboundsArray) = x.a

end # module MakieExt
