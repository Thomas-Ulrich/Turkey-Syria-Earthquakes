import numpy as np
import os
import rasterio
import argparse
import seissolxdmf
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator


def read_observation_data_one_band(fn, downsampling):
    ds = downsampling
    with rasterio.open(fn) as src:
        ew = src.read(1)
        # print("band 1 has shape", ew.shape)
        height, width = ew.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        lon_g, lat_g = rasterio.transform.xy(src.transform, rows, cols)
        lon_g = np.array(lon_g)[::ds, ::ds]
        lat_g = np.array(lat_g)[::ds, ::ds]
        ew = ew[::ds, ::ds]
        return lon_g, lat_g, ew

def compute_triangle_area(xdmfFilename):
    sx = seissolxdmf.seissolxdmf(xdmfFilename)
    geom = sx.ReadGeometry()
    connect = sx.ReadConnect()
    side1 = geom[connect[:, 1], :] - geom[connect[:, 0], :]
    side2 = geom[connect[:, 2], :] - geom[connect[:, 0], :]
    return np.linalg.norm(np.cross(side1, side2), axis=1) / 2
    
def read_seissol_surface_data(xdmfFilename):
    """read unstructured free surface output and associated data.
    compute cell_barycenter"""
    sx = seissolxdmf.seissolxdmf(xdmfFilename)
    xyz = sx.ReadGeometry()
    connect = sx.ReadConnect()
    U = sx.ReadData("u1", sx.ndt - 1)
    V = sx.ReadData("u2", sx.ndt - 1)
    W = sx.ReadData("u3", sx.ndt - 1)

    # project the data to geocentric (lat, lon)

    myproj = "+proj=tmerc +datum=WGS84 +k=0.9996 +lon_0=37.0 +lat_0=37.0"
    transformer = Transformer.from_crs(myproj, "epsg:4326", always_xy=True)
    lons, lats = transformer.transform(xyz[:, 0], xyz[:, 1])
    xy = np.vstack((lons, lats)).T

    # compute triangule barycenter
    lonlat_barycenter = (
        xy[connect[:, 0], :] + xy[connect[:, 1], :] + xy[connect[:, 2], :]
    ) / 3.0

    return lons, lats, lonlat_barycenter, connect, U, V, W


def read_optical_cc(downsampling):
    # Mathilde newest cc results
    fn = "../../ThirdParty/Turquie_detrended_EW_NLM_destripe_wgs84.tif"
    lon_g, lat_g, ew = read_observation_data_one_band(fn, downsampling)
    fn = "../../ThirdParty/Turquie_detrended_NS_NLM_destripe_wgs84.tif"
    lon_g, lat_g, ns = read_observation_data_one_band(fn, downsampling)
    return lon_g, lat_g, ew, ns


def read_scansar(band, downsampling):
    fn = f"../../ThirdParty/Displacement_TUR_20230114_20230207_1529_Data/20230114_HH_20230207_HH.spo_{band}.filtered.geo.tif"
    lon_g, lat_g, obsLOS = read_observation_data_one_band(fn, downsampling)
    obs_to_plot = obsLOS
    fn = "../../ThirdParty/Displacement_TUR_20230114_20230207_1529_Data/20230114_HH_lv_phi.geo.tif"
    lon_g, lat_g, phi_g = read_observation_data_one_band(fn, downsampling)
    fn = "../../ThirdParty/Displacement_TUR_20230114_20230207_1529_Data/20230114_HH_lv_theta.geo.tif"
    lon_g, lat_g, theta_g = read_observation_data_one_band(fn, downsampling)
    obsLOS[obsLOS == 0.0] = np.nan
    return lon_g, lat_g, obsLOS, phi_g, theta_g


def read_insar(band, downsampling):
    id_los = band.split("los")[-1]
    src = rasterio.open(f"../../ThirdParty/InSAR/{id_los}/los_ll_low.tif")
    ds = downsampling
    obs_to_plot = src.read(1)[::ds, ::ds]
    obs_to_plot[obs_to_plot == -32767] = np.nan
    obs_to_plot = obs_to_plot / 1e2
    fn = f"../../ThirdParty/InSAR/{id_los}/vx_ll_low.tif"
    lon_g, lat_g, vx = read_observation_data_one_band(fn, ds)
    fn = f"../../ThirdParty/InSAR/{id_los}/vy_ll_low.tif"
    lon_g, lat_g, vy = read_observation_data_one_band(fn, ds)
    fn = f"../../ThirdParty/InSAR/{id_los}/vz_ll_low.tif"
    lon_g, lat_g, vz = read_observation_data_one_band(fn, ds)
    return lon_g, lat_g, obs_to_plot, vx, vy, vz


def RGIinterp(lon_g, lat_g, data_g, lonlat_eval):
    f = RegularGridInterpolator(
        (lon_g[0, :], lat_g[:, 0]), data_g.T, bounds_error=False, fill_value=np.nan
    )
    return f(lonlat_eval)


def compute_LOS_displacement_SeisSol_data_from_LOS_angles(
    lon_g, lat_g, theta_g, phi_g, lonlat_barycenter, band, U, V, W
):
    # interpolate satellite angles on the unstructured grid
    theta_inter = RGIinterp(lon_g, lat_g, theta_g, lonlat_barycenter)
    phi_inter = RGIinterp(lon_g, lat_g, phi_g, lonlat_barycenter)
    # compute displacement line of sight
    # phi azimuth, theta: range
    if band == "azimuth":
        D_los = U * np.sin(phi_inter) + V * np.cos(phi_inter)
    else:
        D_los = W * np.cos(theta_inter) + np.sin(theta_inter) * (
            U * -np.cos(phi_inter) + V * np.sin(phi_inter)
        )
        # D_los = W * np.sin(theta_inter) + np.cos(theta_inter) * (U * np.cos(phi_inter) + V * np.sin(phi_inter))
    return -D_los


def compute_LOS_displacement_SeisSol_data_from_LOS_vector(
    lon_g, lat_g, vx, vy, vz, lonlat_barycenter, U, V, W
):
    # interpolate satellite LOS on the unstructured grid
    vx_inter = RGIinterp(lon_g, lat_g, vx, lonlat_barycenter)
    vy_inter = RGIinterp(lon_g, lat_g, vy, lonlat_barycenter)
    vz_inter = RGIinterp(lon_g, lat_g, vz, lonlat_barycenter)
    D_los = U * vx_inter + V * vy_inter + W * vz_inter
    return -D_los


def nanrms(x, axis=None):
    return np.sqrt(np.nanmean(x**2, axis=axis))

def nanrms_weighted(x, areas,  axis=None):
    areas[np.isnan(x)] = np.nan
    sum_area = np.nansum(areas)
    return np.sqrt(np.nansum(areas * x**2, axis=axis)/sum_area)
