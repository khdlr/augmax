from .base import BaseChain, Transformation, InputType
from .geometric import GeometricChain, GeometricTransformation
from .colorspace import ColorspaceChain, ColorspaceTransformation

class Chain(BaseChain):
    def __init__(self, *transforms: Transformation, input_types=[InputType.IMAGE]):
        geometric = []
        colorspace = []
        other = []

        sub_chains = []
        for transform in transforms:
            if isinstance(transform, GeometricTransformation):
                if other:
                    sub_chains.append(BaseChain(*other))
                    other = []
                if colorspace:
                    sub_chains.append(ColorspaceChain(*colorspace))
                    colorspace = []
                geometric.append(transform)
            elif isinstance(transform, ColorspaceTransformation):
                if other:
                    sub_chains.append(BaseChain(*other))
                    other = []
                if geometric:
                    sub_chains.append(GeometricChain(*geometric))
                    geometric = []
                colorspace.append(transform)
            else:
                if geometric:
                    sub_chains.append(GeometricChain(*geometric))
                    geometric = []
                if colorspace:
                    sub_chains.append(ColorspaceChain(*colorspace))
                    colorspace = []
                other.append(transform)

        if other:
            sub_chains.append(BaseChain(*other))
            other = []
        if geometric:
            sub_chains.append(GeometricChain(*geometric))
            geometric = []
        if colorspace:
            sub_chains.append(ColorspaceChain(*colorspace))
            colorspace = []

        print(sub_chains)

        super().__init__(*sub_chains, input_types=input_types)


class OptimizedChain(BaseChain):
    def __init__(self, *transforms: Transformation, input_types=[]):
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

        super().__init__(*sub_chains, input_types=input_types)
