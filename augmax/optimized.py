from .base import Chain, Transformation
from .geometric import GeometricChain, GeometricTransformation
from .colorspace import ColorspaceChain, ColorspaceTransformation

class OptimizedChain(Chain):
    def __init__(self, *transforms: Transformation):
        geometric = []
        colorspace = []
        other = []
        for transform in transforms:
            if isinstance(transform, GeometricTransformation):
                geometric.append(transform)
            elif isinstance(transform, ColorspaceTransformation):
                colorspace.append(transform)
            else:
                other.append(transform)

        sub_chains = []
        if geometric:
            sub_chains.append(GeometricChain(*geometric))
        if colorspace:
            sub_chains.append(ColorspaceChain(*colorspace))
        if other:
            sub_chains.append(Chain(*other))

        print(sub_chains)

        super().__init__(*sub_chains)
