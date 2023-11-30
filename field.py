import torch 
import numpy as np
from typing import Dict, Literal, Optional, Tuple
from torch import Tensor, nn

from jaxtyping import Float, Shaped

from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
import tinycudann as tcnn
from my_encodings import get_encoder
from ray_samplers import spacetime_concat


class SpaceTimeHashingField(Field):

    aabb: Tensor    

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 3,
        hidden_dim: int = 128,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 1,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 128,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_linear: bool = False,
        use_appearance_embedding: Optional[bool] = False,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.pass_semantic_gradients = pass_semantic_gradients
        self.base_res = base_res
        self.step = 0        
        self.use_appearance_embedding = use_appearance_embedding
        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )


        self.space_time_grid, self.input_ch_space_time = get_encoder('hash', input_dim=4, n_levels=num_levels, 
                                                                    base_resolution=base_res, desired_resolution=max_res,
                                                                    log2_hashmap_size=log2_hashmap_size, level_dim=features_per_level)
        self.space_grid, self.input_ch_space = get_encoder('hash', input_dim=3, n_levels=num_levels, 
                                                                    base_resolution=base_res, desired_resolution=max_res,
                                                                    log2_hashmap_size=log2_hashmap_size, level_dim=features_per_level) 

        self.mlp_base_mlp = MLP(
            in_dim=self.input_ch_space + self.input_ch_space_time,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        
        in_dim = self.direction_encoding.get_out_dim() + self.geo_feat_dim
        if self.use_appearance_embedding:
            in_dim += self.appearance_embedding_dim
        self.mlp_head = MLP(
            in_dim=in_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    def density_fn(
        self, positions: Shaped[Tensor, "*bs 3"], times: Optional[Shaped[Tensor, "*bs 1"]] = None
    ) -> Shaped[Tensor, "*bs 1"]:
        pos = positions[..., :3]
        times = positions[..., 3:]
        # Need to figure out a better way to describe positions with a ray.
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=pos,
                directions=torch.ones_like(pos),
                starts=torch.zeros_like(pos[..., :1]),
                ends=torch.zeros_like(pos[..., :1]),
                pixel_area=torch.ones_like(pos[..., :1]),
            ),
            times=times,
        )
        density, _ = self.get_density(ray_samples)
        return density
    
    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0 # default the scene box [0,1]
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)        
        

        # concatenate positions and times
        spacetime = spacetime_concat(positions, ray_samples.times).view(-1, 4)
        spacetime_out = self.space_time_grid(spacetime)
        space_out = self.space_grid(positions.view(-1, 3))

        h1 = torch.cat([space_out, spacetime_out], dim=-1)
        h = self.mlp_base_mlp(h1).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        # SH encoding must be in the range [0, 1]
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        
        d = self.direction_encoding(directions_flat)
        outputs_shape = ray_samples.frustums.directions.shape[:-1] 
        if density_embedding is None:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
            h = torch.cat([d, positions.view(-1, 3)], dim=-1)
        else:
            h = torch.cat([d, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)    

        # if self.use_appearance_embedding:
        #     if ray_samples.camera_indices is None:
        #         raise AttributeError("Camera indices are not provided.")
        #     camera_indices = ray_samples.camera_indices.squeeze()
        #     if self.training:
        #         embedded_appearance = self.appearance_embedding(camera_indices)
        #     else:
        #         embedded_appearance = torch.zeros(
        #             (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
        #         )
        #     h = torch.cat([h, embedded_appearance.view(-1, self.appearance_embedding_dim)], dim=-1)

        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs        
    

class SpaceTimeDensityField(Field):
    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear: bool = False,
        num_levels: int = 8,
        max_res: int = 1024,
        base_res: int = 16,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        }

        if not self.use_linear:
            self.mlp_base = tcnn.NetworkWithInputEncoding(
                n_input_dims=4,
                n_output_dims=1,
                encoding_config=config["encoding"],
                network_config=config["network"],
            )
        else:
            self.encoding = tcnn.Encoding(n_input_dims=4, encoding_config=config["encoding"])
            self.linear = torch.nn.Linear(self.encoding.n_output_dims, 1)

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        # note that now this stands for spacetime
        positions_flat = spacetime_concat(positions, ray_samples.times).view(-1, 4)
        # print(positions_flat)
        # positions_flat = positions.view(-1, 4)
        if not self.use_linear:
            density_before_activation = (
                self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1).to(positions)
            )
        else:
            x = self.encoding(positions_flat).to(positions)
            density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        density = density * selector[..., None]
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> dict:
        return {}

    def density_fn(
        self, positions: Shaped[Tensor, "*bs 3"], times: Optional[Shaped[Tensor, "*bs 1"]] = None
    ) -> Shaped[Tensor, "*bs 1"]:        
        pos = positions[..., :3]
        times = positions[..., 3:]
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=pos,
                directions=torch.ones_like(pos),
                starts=torch.zeros_like(pos[..., :1]),
                ends=torch.zeros_like(pos[..., :1]),
                pixel_area=torch.ones_like(pos[..., :1]),
            ),
            times=times,
        )
        density, _ = self.get_density(ray_samples)
        return density


class SpaceTimeDensityFieldWithBase(Field):
    def __init__(
        self,
        aabb: Tensor,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear: bool = False,
        num_levels: int = 8,
        max_res: int = 1024,
        base_res: int = 16,
        log2_hashmap_size: int = 18,
        features_per_level: int = 2,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        }

        if not self.use_linear:
            self.mlp_base = tcnn.NetworkWithInputEncoding(
                n_input_dims=4,
                n_output_dims=1,
                encoding_config=config["encoding"],
                network_config=config["network"],
            )
            self.spatial_mlp_base = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=1,
                encoding_config=config["encoding"],
                network_config=config["network"],
            )
        else:
            self.encoding = tcnn.Encoding(n_input_dims=4, encoding_config=config["encoding"])
            self.linear = torch.nn.Linear(self.encoding.n_output_dims, 1)

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, None]:
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        # note that now this stands for spacetime
        positions_flat = spacetime_concat(positions, ray_samples.times).view(-1, 4)
        # print(positions_flat)
        # positions_flat = positions.view(-1, 4)
        if not self.use_linear:
            density_before_activation = (
                self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1).to(positions) + self.spatial_mlp_base(positions.view(-1, 3)).view(*ray_samples.frustums.shape, -1).to(positions)
            )
        else:
            x = self.encoding(positions_flat).to(positions)
            density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        density = density * selector[..., None]
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> dict:
        return {}

    def density_fn(
        self, positions: Shaped[Tensor, "*bs 3"], times: Optional[Shaped[Tensor, "*bs 1"]] = None
    ) -> Shaped[Tensor, "*bs 1"]:  
        pos = positions[..., :3]
        times = positions[..., 3:]
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=pos,
                directions=torch.ones_like(pos),
                starts=torch.zeros_like(pos[..., :1]),
                ends=torch.zeros_like(pos[..., :1]),
                pixel_area=torch.ones_like(pos[..., :1]),
            ),
            times=times,
        )
        density, _ = self.get_density(ray_samples)
        return density