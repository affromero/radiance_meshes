from utils.train_util import render
from models.vertex_color import Model

render_pkg = render(camera, model, tile_size=tile_size)
path = "/home/dronelab/gaussian-splatting-merge/eval/bicycle/point_cloud/iteration_7000/point_cloud.ply"
plydata = PlyData.read(path)
xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),  axis=1)
opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
features_dc = np.zeros((xyz.shape[0], 3, 1))
features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
