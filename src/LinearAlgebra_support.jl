# Functions from LinearAlgebra.jl need to hand the work back to the optimized
# implementations that work on `Array`, instead of the generic ones for `AbstractArray`.

import LinearAlgebra: lu, lu!, ldiv!, ldiv, Factorization, mul!

@inline function ldiv!(x::InboundsVector, Alu::Factorization, b::InboundsVector)
    ldiv!(x.a, Alu, b.a)
    return x
end
@inline function ldiv!(x::InboundsMatrix, Alu::Factorization, b::InboundsMatrix)
    ldiv!(x.a, Alu, b.a)
    return x
end

@inline function ldiv!(Alu::Factorization, b::InboundsVector)
    ldiv!(Alu, b.a)
    return b
end
@inline function ldiv!(Alu::Factorization, b::InboundsMatrix)
    ldiv!(Alu, b.a)
    return b
end

@inline function ldiv(Alu::Factorization, b::InboundsVector)
    return InboundsArray(ldiv(Alu, b.a))
end
@inline function ldiv(Alu::Factorization, b::InboundsMatrix)
    return InboundsArray(ldiv(Alu, b.a))
end

@inline function lu(m::InboundsMatrix)
    return lu(m.a)
end

@inline function lu!(m::InboundsMatrix)
    return lu(m.a)
end

# Need to use specific enough types for `mul!()` arguments that these versions take
# precedence over AbstractVector versions defined in LinearAlgebra.
@inline function mul!(C::InboundsMatrix, A::InboundsMatrix, B::InboundsMatrix, α::Number, β::Number)
    mul!(C.a, A.a, B.a, α, β)
    return C
end
@inline function mul!(C::InboundsVector, A::InboundsMatrix, B::InboundsVector, α::Number, β::Number)
    mul!(C.a, A.a, B.a, α, β)
    return C
end
