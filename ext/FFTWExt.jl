module FFTWExt

using InboundsArrays

import FFTW
import FFTW: plan_fft!, plan_ifft!
import Base: *

@inline function plan_fft!(a::AbstractInboundsArray, args...; kwargs...)
    return plan_fft!(a.a, args...; kwargs...)
end

@inline function plan_ifft!(a::AbstractInboundsArray, args...; kwargs...)
    return plan_ifft!(a.a, args...; kwargs...)
end

@inline function *(p::FFTW.FFTWPlan, a::InboundsArray)
    return InboundsArray(*(p, a.a))
end
@inline function *(p::FFTW.DCTPlan, a::InboundsArray)
    return InboundsArray(*(p, a.a))
end
@inline function *(p::FFTW.AbstractFFTs.AdjointPlan, a::InboundsArray)
    return InboundsArray(*(p, a.a))
end
@inline function *(p::FFTW.ScaledPlan, a::InboundsArray)
    return InboundsArray(*(p, a.a))
end

end # module FFTWExt
