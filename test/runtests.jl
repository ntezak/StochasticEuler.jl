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

function test_ieuler_sde()
    a, b, c, d = [1,2,3,4,7]

  function linear_sde_ito!(t, x, xdot, gdW, dW, compute_xdot, compute_gdW)
      if compute_xdot
          xdot[1]=(a*x[1] + c)
      end
      if compute_gdW
          gdW[1] = (b*x[1] + d) * dW[1]
      end
      xdot, gdW
  end

  function linear_sde_strat!(t, x, xdot, gdW, dW, compute_xdot, compute_gdW)
      if compute_xdot
          xdot[1]=((a-.5*b^2)*x[1] + c - .5*b*d)
      end
      if compute_gdW
          gdW[1] = (b*x[1]+d)*dW[1]
      end
      xdot, gdW
  end


  t0=0
  t1=.5
  ts = linspace(t0, t1, 1001)
  h = 1./(2<<17)
  x0 = [1.]
  println("Running Stratonovich SDE integration")
  @time ts, xs1, dWs, iters1 = ieuler_heun(linear_sde_strat!, x0, ts, h, 1; return_metrics=true, ϵ_rel=1e-10, max_iter=20)
  println("Running Ito SDE integration")
  @time ts, xs2, dWs, iters2 = ieuler_mayurama(linear_sde_ito!, x0, ts, h, 1; return_metrics=true, ϵ_rel=1e-10, max_iter=20)

  dts = diff(ts)
  dWs = dWs'
  Φtt0 = exp(cumsum((a-.5*b^2) * dts + b*dWs))
  xtsol = [x0[1]; Φtt0 .* (x0[1] + cumsum((c) *dts ./ Φtt0)  + cumsum(d * dWs ./ Φtt0))]

  rel_err1 = abs(xs1'-xtsol) ./ (abs(xtsol)+abs(xs1'))
  rel_err2 = abs(xs2'-xtsol) ./ (abs(xtsol)+abs(xs2'))
  @test mean(rel_err1[500:end]) < 1e-2
  @test mean(rel_err2[500:end]) < 1e-2
end


test_cumulative_normal()
test_ieuler_sde()
