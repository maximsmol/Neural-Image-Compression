from PIL import Image, ImageCms

srgb_profile = ImageCms.createProfile("sRGB")
lab_profile = ImageCms.createProfile("LAB")

class CIELABTransform(object):
  def __init__(self):
    self.t = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")

  def __call__(self, img):
    if img.mode != "RGB":
      img = img.convert("RGB")

    return ImageCms.applyTransform(img, self.t)

class InvCIELABTransform(object):
  def __init__(self):
    self.t = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB", "RGB")

  def __call__(self, img):
    if img.mode != "LAB":
      img = img.convert("LAB")

    return ImageCms.applyTransform(img, self.t)
