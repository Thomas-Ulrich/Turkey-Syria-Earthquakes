import numpy as np
a = np.loadtxt('../ThirdParty/vel1d_Guvercin_et_al.txt', skiprows=0)
n = int(a.shape[0]//3)
a = a.reshape((n,3))
a[0,0] = -10
a[-1,0] = 1000
print(a)

towrite_luamap = """!LuaMap
returns: [rho, mu, lambda, plastCo]
function: |
  function f (x)
    xi = x["z"]
    aX = {}
"""

towrite_luamap_end = """    return {
     rho = rho,
     mu = mu,
     lambda = lambda,
     plastCo = 0.0004 * mu
    }
    end
"""

towrite = """!LayeredModel
  map: !AffineMap
    matrix:
      z1: [0.0, 0.0, 1.0]
    translation:
      z1: 0
  interpolation: lower
  parameters: [rho, mu, lambda, plastCo]
  nodes:
"""
def vsrhomulambda(VP, VS):
    rho = 1.6612*VP-0.4721*VP**2+0.0671*VP**3-0.0043*VP**4+0.000106*VP**5
    G = rho*VS*VS
    lambdax = 1e9*rho*(VP**2-2.*VS**2)
    rho = 1e3*rho
    mu = 1e9*G
    return VS, rho, mu, lambdax


nrows = a.shape[0]
aData = np.zeros((nrows, 4))
for i, row in enumerate(a):
    depth, vp, vs = row
    vs, rho, mu, lambdax = vsrhomulambda(vp, vs)
    print(-depth*1e3, mu)
    aData[i,:] = -depth*1e3, rho, mu, lambdax
    towrite += f"      {-depth*1000}: [{rho}, {mu}, {lambdax}, {mu*0.0004}]\n"

aData = np.flip(aData, axis=0)

for i, row in enumerate(aData):
    towrite_luamap += f"    aX[{i+1}] = " + "{" + ",".join([str(val) for val in aData[i,:]])  + "}\n"

towrite_luamap += """
    if (aX[1][1] > xi) or ( xi>aX[#aX][1] ) then
      io.write(aX[1][1], " ", xi, " ", aX[#aX][1], " are not sorted as expected\\n")
    end

"""

towrite_luamap += """    for i in pairs(aX) do
      if (aX[i][1]>xi) then
        rho = aX[i-1][2]
        mu = aX[i-1][3]
        lambda = aX[i-1][4]
        break
      end
    end
"""


fn = 'Mw_78_Turkey_rhomulambda1D_Guvercin_et_al.yaml'
with open(fn, 'w') as fid:
    fid.write(towrite)
print(f'done writing {fn}')
fn = 'Mw_78_Turkey_rhomulambda1D_Guvercin_et_al_lua.yaml'
with open(fn, 'w') as fid:
    fid.write(towrite_luamap)
    fid.write(towrite_luamap_end)
print(f'done writing {fn}')
