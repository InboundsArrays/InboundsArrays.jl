module NaNMathExt

using InboundsArrays

import NaNMath

const oneargfuncs = (
    :argmax, :argmin, :extrema, :findmax, :findmin, :maximum, :mean, :mean_count, :median,
    :minimum, :std, :sum, :var
)

for func ∈ oneargfuncs
    eval(quote
        @inline function NaNMath.$func(a::AbstractInboundsArray)
            return InboundsArray(NaNMath.$func(a.a))
        end
    end)
end

end # module NaNMathExt
