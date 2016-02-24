using StochasticEuler
using Base.Test

function test_cumulative_normal()
    N = 20
    M = 30
    s = 0
    crng = CumulativeNormal(N,M)
    srand(crng, s)
    cdata = zeros(M)
    @time randn!(crng, cdata)


    data = zeros(M, N)
    srand(crng.rng, s)
    @time randn!(crng.rng, data)

    @test norm(sum(data, 2)/sqrt(N) - cdata) < 1e-8

    srand(crng, s)
    cdata = zeros(Complex128, M)
    @time randn!(crng, cdata)

    data = zeros(Float64, 2,M,N)
    srand(crng.rng, s)
    @time randn!(crng.rng, data)
    data = reshape(data[1,:,:] + 1im * data[2,:,:], (M,N))
    @test norm(sum(data, 2)/sqrt(N) - cdata) < 1e-8

end

test_cumulative_normal()
