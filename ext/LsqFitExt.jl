module LsqFitExt

using InboundsArrays

import LsqFit: curve_fit

curve_fit(model, jacobian_model, xdata::AbstractInboundsArray, ydata::AbstractInboundsArray, wt::AbstractInboundsArray{T, 2} where T, p0::AbstractInboundsArray; kwargs...) =
    curve_fit(model, jacobian_model, xdata.a, ydata.a, wt.a, p0.a; kwargs...)
curve_fit(model, xdata::AbstractInboundsArray, ydata::AbstractInboundsArray, wt::AbstractInboundsArray{T, 2} where T, p0::AbstractInboundsArray; kwargs...) =
    curve_fit(model, xdata.a, ydata.a, wt.a, p0.a; kwargs...)
curve_fit(model, jacobian_model, xdata::AbstractInboundsArray, ydata::AbstractInboundsArray, p0::AbstractInboundsArray; kwargs...) =
    curve_fit(model, jacobian_model, xdata.a, ydata.a, p0.a; kwargs...)
curve_fit(model, xdata::AbstractInboundsArray, ydata::AbstractInboundsArray, p0::AbstractInboundsArray; kwargs...) =
    curve_fit(model, xdata.a, ydata.a, p0.a; kwargs...)

end # module LsqFitExt
