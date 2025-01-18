module InboundsArraysTests

using InboundsArrays
using Test

function isequal(args...)
    return isapprox(args...; rtol=0.0, atol=0.0)
end

function runtests()

    @testset "InboundsArrays" begin
        @testset "InboundsVector" begin
            a = InboundsVector([1.0, 2.0, 3.0, 4.0])
            b = InboundsVector([5.0, 6.0, 7.0, 8.0])

            @test size(a) == (4,)
            @test length(a) == 4
            @test a[2] == 2.0
            a[2] = 42.0
            @test a[2] == 42.0
            a[2] = 2.0

            for i ∈ 1:length(a)
                a[i] += b[i]
            end

            @test isequal(a, [6.0, 8.0, 10.0, 12.0])

            c = a .+ b
            @test c isa InboundsVector
            @test isequal(c, [11.0, 14.0, 17.0, 20.0])

            c .= a .* b
            @test c isa InboundsVector
            @test isequal(c, [30.0, 48.0, 70.0, 96.0])

            d = similar(a)
            @test d isa InboundsVector{Float64, Vector{Float64}}
            @test size(d) == (4,)

            d = similar(a, Int64)
            @test d isa InboundsVector{Int64, Vector{Int64}}
            @test size(d) == (4,)

            d = similar(a, (3, 5))
            @test d isa InboundsArray{Float64, 2, Array{Float64, 2}}
            @test size(d) == (3, 5)

            d = similar(a, Int64, (3, 5))
            @test d isa InboundsArray{Int64, 2, Array{Int64, 2}}
            @test size(d) == (3, 5)

            @test axes(a) == (1:4,)
        end

        @testset "InboundsMatrix" begin
            a = InboundsMatrix([1.0 2.0; 3.0 4.0])
            b = InboundsMatrix([5.0 6.0; 7.0 8.0])

            @test size(a) == (2, 2)
            @test length(a) == 4
            @test a[1, 2] == 2.0
            a[1, 2] = 42.0
            @test a[1, 2] == 42.0
            a[1, 2] = 2.0

            for i ∈ 1:length(a)
                a[i] += b[i]
            end

            @test isequal(a, [6.0 8.0; 10.0 12.0])

            a .= [1.0 2.0; 3.0 4.0]

            for i ∈ 1:size(a, 2), j ∈ 1:size(a, 1)
                a[j, i] += b[j, i]
            end

            @test isequal(a, [6.0 8.0; 10.0 12.0])

            c = a .+ b
            @test c isa InboundsMatrix
            @test isequal(c, [11.0 14.0; 17.0 20.0])

            c .= a .* b
            @test c isa InboundsMatrix
            @test isequal(c, [30.0 48.0; 70.0 96.0])

            d = similar(a)
            @test d isa InboundsMatrix{Float64, Matrix{Float64}}
            @test size(d) == (2, 2)

            d = similar(a, Int64)
            @test d isa InboundsMatrix{Int64, Matrix{Int64}}
            @test size(d) == (2, 2)

            d = similar(a, (3, 5, 6))
            @test d isa InboundsArray{Float64, 3, Array{Float64, 3}}
            @test size(d) == (3, 5, 6)

            d = similar(a, Int64, (3, 5, 6))
            @test d isa InboundsArray{Int64, 3, Array{Int64, 3}}
            @test size(d) == (3, 5, 6)

            @test axes(a) == (1:2, 1:2)
        end

        @testset "InboundsArray" begin
            a = InboundsArray([1.0 2.0; 3.0 4.0;;; 1.0 2.0; 3.0 4.0])
            b = InboundsArray([5.0 6.0; 7.0 8.0;;; 5.0 6.0; 7.0 8.0])

            @test size(a) == (2, 2, 2)
            @test length(a) == 8
            @test a[1, 2, 1] == 2.0
            a[1, 2, 1] = 42.0
            @test a[1, 2, 1] == 42.0
            a[1, 2, 1] = 2.0

            for i ∈ 1:length(a)
                a[i] += b[i]
            end

            @test isequal(a, [6.0 8.0; 10.0 12.0;;; 6.0 8.0; 10.0 12.0])

            a .= [1.0 2.0; 3.0 4.0;;; 1.0 2.0; 3.0 4.0]

            for i ∈ 1:size(a, 3), j ∈ 1:size(a, 2), k ∈ 1:size(a, 1)
                a[k, j, i] += b[k, j, i]
            end

            @test isequal(a, [6.0 8.0; 10.0 12.0;;; 6.0 8.0; 10.0 12.0])

            c = a .+ b
            @test c isa InboundsArray
            @test isequal(c, [11.0 14.0; 17.0 20.0;;; 11.0 14.0; 17.0 20.0])

            c .= a .* b
            @test c isa InboundsArray
            @test isequal(c, [30.0 48.0; 70.0 96.0;;; 30.0 48.0; 70.0 96.0])

            c = a .+ 2
            @test c isa InboundsArray
            @test isequal(c, [8.0 10.0; 12.0 14.0;;; 8.0 10.0; 12.0 14.0])

            d = similar(a)
            @test d isa InboundsArray{Float64, 3, Array{Float64, 3}}
            @test size(d) == (2, 2, 2)

            d = similar(a, Int64)
            @test d isa InboundsArray{Int64, 3, Array{Int64, 3}}
            @test size(d) == (2, 2, 2)

            d = similar(a, (3, 5))
            @test d isa InboundsArray{Float64, 2, Array{Float64, 2}}
            @test size(d) == (3, 5)

            d = similar(a, Int64, (3, 5))
            @test d isa InboundsArray{Int64, 2, Array{Int64, 2}}
            @test size(d) == (3, 5)

            @test axes(a) == (1:2, 1:2, 1:2)
        end
    end

end

end # module InboundsArraysTests

using .InboundsArraysTests
InboundsArraysTests.runtests()
