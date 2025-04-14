import numpy as np
from mayavi import mlab
import nibabel as nib

from tvtk.util.ctf import ColorTransferFunction, PiecewiseFunction

filename = "/Users/zhangenzhi/Desktop/zez/samples/s8d/masked_image_34.raw"

import numpy as np
from mayavi import mlab

# Assuming your CT data is loaded in a numpy array called 'ct_data'
# ct_data = np.load('your_data.npy')  # or however you load your data

# For demonstration, let's create some synthetic CT-like data
# ct_data = np.random.rand(160, 512, 512) * 1000  # Replace this with your actual data

# filename = "ct_scan.raw"
dtype = np.float32  # float32 data
shape = (160, 512, 512)  # Depth, height, width (adjust as needed)
byte_order = 'little'  # or 'big' (check your data source)

# Read binary data
with open(filename, 'rb') as f:
    ct_data = np.fromfile(f, dtype=dtype)

# Reshape and handle byte order if necessary
ct_data = ct_data.reshape(shape)
if byte_order != 'little':  # Most systems use little-endian
    ct_data = ct_data.byteswap().newbyteorder()

# 2. 归一化（保留0值）
data_min, data_max = ct_data.min(), ct_data.max()
ct_data_normalized = (ct_data - data_min) / (data_max - data_min)

# 3. 创建Mayavi场景
fig = mlab.figure(size=(1200, 900), bgcolor=(0, 0, 0))  # 黑色背景更清晰

# 4. 创建体渲染
vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(ct_data_normalized))

# 5. 关键步骤：设置透明度传输函数（让0值完全透明）
otf = PiecewiseFunction()
otf.add_point(0.0, 0.0)    # 值为0 → 完全透明
otf.add_point(0.01, 0.0)   # 接近0的值也透明（避免浮点误差）
otf.add_point(0.1, 0.02)   # 低值半透明
otf.add_point(0.5, 0.1)    # 中等值
otf.add_point(1.0, 0.3)    # 高值更不透明

# 6. 颜色传输函数（灰度显示）
ctf = ColorTransferFunction()
ctf.add_rgb_point(0.0, 0.0, 0.0, 0.0)  # 0值颜色（虽然透明但仍需定义）
ctf.add_rgb_point(0.5, 0.5, 0.5, 0.5)  # 中间灰度
ctf.add_rgb_point(1.0, 1.0, 1.0, 1.0)  # 白色

# 7. 应用属性
vol_property = vol.volume_property
vol_property.set_color(ctf)
vol_property.set_scalar_opacity(otf)
vol_property.shade = True  # 启用光照

# 8. 优化显示
mlab.view(azimuth=45, elevation=60, distance=600)
mlab.outline()  # 添加边框辅助定位
mlab.show()