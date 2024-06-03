from monai.transforms import MapTransform

class ConvertLabelBasedOnClasses(MapTransform):
    def __call__(self, data):
        d = dict(data)
        # print(data)
        for key in self.keys:
            result = d[key] > 1
            d[key] = result.float()
        return d