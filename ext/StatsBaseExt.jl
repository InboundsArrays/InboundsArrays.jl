module StatsBaseExt

using InboundsArrays

import StatsBase

const oneargfuncs = (
    :autocor, :autocov, :aweights, :competerank, :cor, :corkendall, :corspearman, :counts,
    :countties, :cov, :cov2cor, :cronbachalpha, :cumulant, :denserank, :durbin, :ecdf,
    :eweights, :fweights, :genvar, :histrange, :indexmap, :indicatormat, :insertion_sort!,
    :kurtosis, :levelsmap, :mad!, :mean, :mean_and_std, :mean_and_var, :median, :median!,
    :merge_sort!, :middle, :midpoints, :mode, :modes, :moment, :norepeats, :ordinalrank,
    :proportionmap, :proportions, :pweights, :quantile, :quantile!, :renyientropy, :rle,
    :sem, :skewness, :std, :summarystats, :tiedrank, :totalvar, :trim, :trim!, :trimvar,
    :uplo, :var, :weights, :winsor, :winsor!, :wmedian, :wquantile, :wsum, :zscore,
    :zscore!,
)

const twoargfuncs = (
    :L1dist, :L2dist, :Linfdist, :addcounts!, :autocor, :autocov, :cor, :cor2cov,
    :cor2cov!, :corkendall, :corkendall!, :corspearman, :counteq, :countmap, :countne,
    :counts, :cov, :cov2cor, :crosscor, :crosscov, :crossentropy, :demean_col!, :durbin!,
    :fisher_yates_sample!, :gkldiv, :indicatormat, :inverse_rle, :kldivergence,
    :knuths_sample!, :levinson, :maxad, :mean!, :meanad, :msd, :pacf, :proportions, :psnr,
    :quantile!, :rmsd, :sqL2dist, :var!, :wmean, :wmedian, :wquantile, :wsum, :zscore!,
)

const threeargfuncs = (
    :addcounts!, :autocor!, :autocov!, :corkendall!, :crosscor, :crosscov, :levinson!,
    :pacf!, :pacf_regress!, :pacf_yulewalker!, :partialcor, :quantile!, :wsum!, :zscore,
    :zscore!,
)

const fourargfuncs = (
    :crosscor!, :crosscov!, :zscore!,
)

for func ∈ oneargfuncs
    eval(quote
        @inline function StatsBase.$func(a::AbstractInboundsArray, args...; kwargs...)
            return InboundsArray(StatsBase.$func(a.a, args...; kwargs...))
        end
    end)
end

@inline function StatsBase.countties(a::AbstractInboundsArray, lo::Integer, hi::Integer)
    return InboundsArray(StatsBase.countties(a.a, lo, hi))
end

@inline function StatsBase.cumulant(a::InboundsArrays.AbstractInboundsArray{<:Real}, i::Union{Integer, AbstractRange{<:Integer}})
    return InboundsArray(StatsBase.cumulant(a.a, i))
end

@inline function StatsBase.eweights(a::InboundsArrays.AbstractInboundsArray{T, 1} where T, r::AbstractRange, x::Real)
    return InboundsArray(StatsBase.eweights(a.a, r, x))
end

@inline function StatsBase.histrange(a::AbstractInboundsArray, i::Integer)
    return InboundsArray(StatsBase.histrange(a.a, i))
end

@inline function StatsBase.insertion_sort!(a::AbstractInboundsArray, i::Integer, j::Integer)
    return InboundsArray(StatsBase.insertion_sort!(a.a, i, j))
end

@inline function StatsBase.merge_sort!(a::AbstractInboundsArray, i::Integer, j::Integer)
    return InboundsArray(StatsBase.merge_sort!(a.a, i, j))
end

@inline function StatsBase.moment(a::AbstractInboundsArray{<:Real}, i::Int64)
    return InboundsArray(StatsBase.moment(a.a, i))
end

@inline function StatsBase.quantile!(a::AbstractInboundsArray{T, 1} where T, t::Tuple{Vararg{Real}}; kwargs...)
    return InboundsArray(StatsBase.quantile!(a.a, t); kwargs...)
end

@inline function StatsBase.renyientropy(a::AbstractInboundsArray, x::Real)
    return InboundsArray(StatsBase.renyientropy(a.a, x))
end

@inline function StatsBase.wmedian(a::AbstractInboundsArray, x::StatsBase.AbstractWeights)
    return InboundsArray(StatsBase.wmedian(a.a, x))
end

@inline function StatsBase.wquantile(a::AbstractInboundsArray, w::StatsBase.AbstractWeights, x::Number)
    return InboundsArray(StatsBase.wquantile(a.a, w, x))
end

@inline function StatsBase.wsum(a::AbstractInboundsArray, w::StatsBase.Weights, dims::Colon)
    return InboundsArray(StatsBase.wsum(a.a, w, dims))
end

@inline function StatsBase.zscore!(a::AbstractInboundsArray, x::Real, y::Real)
    return InboundsArray(StatsBase.zscore!(a.a, x, y))
end

for func ∈ twoargfuncs
    eval(quote
        @inline function StatsBase.$func(a::AbstractInboundsArray, b::AbstractInboundsArray, args...; kwargs...)
            return InboundsArray(StatsBase.$func(a.a, b.a, args...; kwargs...))
        end
    end)
end

@inline function StatsBase.addcounts!(a::AbstractInboundsArray, b::AbstractInboundsArray{<:Integer}, c::UnitRange{<:Integer})
    return InboundsArray(StatsBase.addcounts!(a.a, b.a, c))
end

@inline function StatsBase.demean_col!(a::AbstractInboundsArray{<:Real, 1}, b::AbstractInboundsArray{<:Real, 2}, c::Int64, d::Bool)
    return InboundsArray(StatsBase.demean_col!(a.a, b.a, c, d))
end

@inline function StatsBase.psnr(a::AbstractInboundsArray, b::AbstractInboundsArray, c::Real)
    return InboundsArray(StatsBase.psnr(a.a, b.a, c))
end

@inline function StatsBase.var!(a::AbstractInboundsArray, b::AbstractInboundsArray{<:Real}, c::StatsBase.AbstractWeights, d::Int64; kwargs...)
    return InboundsArray(StatsBase.var!(a.a, b.a, c, d; kwargs...))
end

@inline function StatsBase.wquantile(a::AbstractInboundsArray{<:Real, 1}, b::AbstractInboundsArray{<:Real, 1}, c::Number)
    return InboundsArray(StatsBase.wquantile(a.a, b.a, c))
end

@inline function StatsBase.zscore!(a::AbstractInboundsArray{<:AbstractFloat}, b::AbstractInboundsArray{<:Real}, c::Real, d::Real)
    return InboundsArray(StatsBase.zscore!(a.a, b.a, c, d))
end

for func ∈ threeargfuncs
    eval(quote
        @inline function StatsBase.$func(a::AbstractInboundsArray, b::AbstractInboundsArray, c::AbstractInboundsArray, args...; kwargs...)
            return InboundsArray(StatsBase.$func(a.a, b.a, c.a, args...; kwargs...))
        end
    end)
end

@inline function StatsBase.addcounts!(a::AbstractInboundsArray, b::AbstractInboundsArray{<:Integer}, c::AbstractInboundsArray{<:Integer}, d::Tuple{UnitRange{<:Integer}, UnitRange{<:Integer}})
    return InboundsArray(StatsBase.addcounts!(a.a, b.a, c.a, d))
end

@inline function StatsBase.pacf_regress!(a::AbstractInboundsArray{<:Real, 2}, b::AbstractInboundsArray{<:Real, 2}, c::AbstractInboundsArray{<:Integer, 1}, d::Integer)
    return InboundsArray(StatsBase.pacf_regress!(a.a, b.a, c.a, d))
end

@inline function StatsBase.pacf_yulewalker!(a::AbstractInboundsArray{<:Real, 2}, b::AbstractInboundsArray{T, 2} where T, c::AbstractInboundsArray{<:Integer, 1}, d::Integer)
    return InboundsArray(StatsBase.pacf_yulewalker!(a.a, b.a, c.a, d))
end

@inline function StatsBase.wsum!(a::AbstractInboundsArray, b::AbstractInboundsArray{T}, c::AbstractInboundsArray{T, 1}, d::Int64) where T
    return InboundsArray(StatsBase.wsum!(a.a, b.a, c.a, d))
end

for func ∈ fourargfuncs
    eval(quote
        @inline function StatsBase.$func(a::AbstractInboundsArray, b::AbstractInboundsArray, c::AbstractInboundsArray, d::AbstractInboundsArray, args...; kwargs...)
            return InboundsArray(StatsBase.$func(a.a, b.a, c.a, d.a, args...; kwargs...))
        end
    end)
end

end # module StatsBaseExt
