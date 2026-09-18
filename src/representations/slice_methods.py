from omegaconf import ListConfig
import torch
from typing import Tuple, List, Optional
from src.representations.base_coord_based_representation import CoordBasedRepresentation
from src.representations.mesh import SliceableMesh
import math

from abc import ABC, abstractmethod


class SliceMethod(ABC):
    @abstractmethod
    def __call__(
        self, volume: torch.Tensor, outer_iteration: int, inner_iteration: int
    ) -> List[torch.Tensor]:
        pass


class RandomAverageSlabsWithSliceableMesh(SliceMethod):
    def __init__(
        self,
        slice_budget: int = 5,
        slab_thickness: int = 3,
        slice_stride: int = 0,
        volume_indices: Tuple[int, int, int] = [2, 3, 4],  # Z, Y, X
        slices_discrete: bool = False,
        random_downsample_mesh: bool = False,
        random_downsample_mesh_range: List[float] = [0.1, 1.0],
        grid_aligned: Optional[bool] = None,
        slice_enabled: Tuple[bool, bool, bool] = [True, True, True],
        average_slabs: Tuple[bool, bool, bool] = [False, False, False],
        swapaxis: Tuple[bool, bool, bool] = [False, False, False],
        rnd_indices: Tuple[bool, bool, bool] = [False, False, False],
        keep_dims: Tuple[bool, bool, bool] = [False, False, False],
        stack_num_slices: int = 1,
        stack_padding_mode: str = "edge",
        selection_strategy: Optional[str] = None,
        alternating_slice_enabled: Optional[Tuple[bool, bool, bool]] = None,
    ):

        self.slice_budget = slice_budget
        self.slices_discrete = slices_discrete
        self.slab_thickness = slab_thickness
        self.grid_aligned = grid_aligned
        self.stride = slice_stride
        self.volume_indices = volume_indices
        if isinstance(slice_enabled, str):
            slice_enabled = tuple(s.strip().lower() == "true" for s in slice_enabled.split(","))
        elif isinstance(slice_enabled, ListConfig):
            slice_enabled = tuple(bool(s) for s in slice_enabled)
        else:
            slice_enabled = tuple(slice_enabled)

        if alternating_slice_enabled is None:
            alternating_slice_enabled = slice_enabled
        elif isinstance(alternating_slice_enabled, str):
            alternating_slice_enabled = tuple(
                s.strip().lower() == "true"
                for s in alternating_slice_enabled.split(",")
            )
        elif isinstance(alternating_slice_enabled, ListConfig):
            alternating_slice_enabled = tuple(bool(s) for s in alternating_slice_enabled)
        else:
            alternating_slice_enabled = tuple(alternating_slice_enabled)

        self.slice_enabled = slice_enabled
        self.alternating_slice_enabled = alternating_slice_enabled
        self.average_slabs = average_slabs
        self.swapaxis = swapaxis
        self.rnd_indices = rnd_indices
        self.keep_dims = keep_dims
        self.random_downsample_mesh = random_downsample_mesh
        self.random_downsample_mesh_range = random_downsample_mesh_range
        self.stack_num_slices = stack_num_slices
        self.stack_padding_mode = stack_padding_mode
        self.selection_strategy = selection_strategy

        if self.stack_num_slices < 1:
            raise ValueError(
                f"stack_num_slices must be a positive integer, got {self.stack_num_slices}."
            )
        if self.stack_padding_mode not in ("edge",):
            raise ValueError(
                f"Unsupported stack_padding_mode '{self.stack_padding_mode}'. Currently supported: ['edge']."
            )
        if self.selection_strategy in ("None", "none", "null"):
            self.selection_strategy = None
        if self.selection_strategy not in (
            None,
            "random_single_slices",
            "random_slabs",
            "deterministic_blocks",
            "deterministic_blocks_alternating_axes",
        ):
            raise ValueError(
                "selection_strategy must be one of None, 'random_single_slices', "
                "'random_slabs', 'deterministic_blocks', or "
                "'deterministic_blocks_alternating_axes'."
            )
        if self.selection_strategy is not None and not self.slices_discrete:
            raise ValueError("selection_strategy currently requires slices_discrete=True.")
        if self.slice_budget < 1:
            raise ValueError(f"slice_budget must be positive, got {self.slice_budget}.")
        if self.slab_thickness < 1:
            raise ValueError(
                f"slab_thickness must be positive, got {self.slab_thickness}."
            )

    def __call__(
        self,
        representation: CoordBasedRepresentation,
        mesh: SliceableMesh,
        mesh_data: SliceableMesh,
        outer_iteration: int,
        inner_iteration: int,
    ) -> List[torch.Tensor]:

        slice_budget = self.slice_budget

        ret = []
        ret_slice_inds = []
        strategy_slice_enabled = (
            self.alternating_slice_enabled
            if self.selection_strategy == "deterministic_blocks_alternating_axes"
            else self.slice_enabled
        )
        enabled_axis_positions = [
            i for i, enabled in enumerate(strategy_slice_enabled) if enabled
        ]
        alternating_axis_position = None
        alternating_block_iteration = outer_iteration
        if self.selection_strategy == "deterministic_blocks_alternating_axes":
            if len(enabled_axis_positions) == 0:
                return ret, ret_slice_inds
            alternating_axis_position = enabled_axis_positions[
                outer_iteration % len(enabled_axis_positions)
            ]
            alternating_block_iteration = outer_iteration // len(enabled_axis_positions)

        for (
            axis_position,
            (
                volume_index,
                slice_enabled,
                averageing,
                swapping,
                rnd_index,
                keep_dim,
            ),
        ) in enumerate(zip(
            self.volume_indices,
            self.slice_enabled,
            self.average_slabs,
            self.swapaxis,
            self.rnd_indices,
            self.keep_dims,
        )):
            if self.selection_strategy == "deterministic_blocks_alternating_axes":
                slice_enabled = strategy_slice_enabled[axis_position]
            if (
                self.selection_strategy == "deterministic_blocks_alternating_axes"
                and axis_position != alternating_axis_position
            ):
                continue
            if slice_enabled:
                slab_thickness = self.slab_thickness
                rep = math.ceil(float(slice_budget) / slab_thickness)
                if self.selection_strategy == "random_single_slices":
                    slab_thickness = 1
                    rep = slice_budget
                    rnd_index = True
                    averageing = False
                elif self.selection_strategy == "random_slabs":
                    rnd_index = True
                    averageing = False
                elif self.selection_strategy in (
                    "deterministic_blocks",
                    "deterministic_blocks_alternating_axes",
                ):
                    slab_thickness = slice_budget
                    rep = 1
                    rnd_index = False
                    averageing = False

                if self.slices_discrete:

                    slice_dim = mesh.matrix_size[volume_index]

                    stride = self.stride
                    if self.selection_strategy in (
                        "deterministic_blocks",
                        "deterministic_blocks_alternating_axes",
                    ):
                        assert (
                            stride == 0
                        ), "Deterministic block cycling requires stride == 0"
                        num_blocks = math.ceil(float(slice_dim) / slice_budget)
                        block_iteration = (
                            alternating_block_iteration
                            if self.selection_strategy == "deterministic_blocks_alternating_axes"
                            else outer_iteration
                        )
                        block_index = block_iteration % num_blocks
                        start = block_index * slice_budget
                        stop = min(start + slice_budget, slice_dim)
                        slice_inds = torch.arange(
                            start, stop, device=representation.device
                        )
                    else:
                        index_mask = (
                            (stride + 1)
                            * (
                                torch.arange(
                                    slab_thickness, device=representation.device
                                )
                                - slab_thickness // 2
                            )
                        ).repeat(rep)
                        offset = (stride + 1) * (slab_thickness // 2)

                    if (
                        self.selection_strategy not in (
                            "deterministic_blocks",
                            "deterministic_blocks_alternating_axes",
                        )
                        and rnd_index
                    ):
                        slice_inds = (
                            torch.randint(
                                offset,
                                slice_dim - offset,
                                (rep,),
                                device=representation.device,
                            ).repeat_interleave(slab_thickness)
                            + index_mask
                        )

                        slice_inds = slice_inds[:slice_budget]
                    elif self.selection_strategy not in (
                        "deterministic_blocks",
                        "deterministic_blocks_alternating_axes",
                    ):
                        assert (
                            stride == 0
                        ), "Cycling mode not yet supported with stride > 0"
                        slice_inds = (
                            (
                                offset
                                + inner_iteration * (stride + 1) * slab_thickness
                            )
                            * torch.ones((1,), device=representation.device)
                        ).repeat_interleave(slab_thickness) + index_mask
                        slice_inds = slice_inds[slice_inds < slice_dim]

                    slices = representation.forward(
                        mesh.add_index_select(volume_index, slice_inds.int())
                    )
                    ret_slice_inds.append(slice_inds)
                # else:
                    # assert (
                        # rnd_index
                    # ), "Continuous slicing only supported with random indices"

                    # if isinstance(self.stride, ListConfig):
                        # stride = self.stride[0] + torch.rand(
                            # (1,), device=representation.device
                        # ).item() * (self.stride[1] - self.stride[0])
                    # else:
                        # stride = self.stride

                    # lb, ub = (
                        # mesh.lower_coords[volume_index],
                        # mesh.upper_coords[volume_index],
                    # )  # e.g. -1.0, 1.0

                    # assert (
                        # self.grid_aligned is not None
                    # ), "grid_aligned must be set for continuous slicing"

                    # grid_aligned = self.grid_aligned

                    # stepsize = (ub - lb) / (mesh.matrix_size[volume_index] - 1)
                    # if grid_aligned:
                        # pix_start = torch.randint(
                            # 0,
                            # mesh.matrix_size[volume_index] - self.slab_thickness,
                            # (rep,),
                            # device=representation.device,
                        # )
                        # slices_lb = lb + stepsize * pix_start
                    # else:
                        # ub_red = max(
                            # ub - (self.slab_thickness - 1) * stepsize * stride, lb
                        # )
                        # slices_lb = lb + (ub_red - lb) * torch.rand(
                            # (rep,), device=representation.device
                        # )  # shape (rep,) #

                    # slices_ub = (
                        # slices_lb + (self.slab_thickness - 1) * stepsize * stride
                    # )  # shape (rep,)

                    # random_downsample_mesh = self.random_downsample_mesh
                    # random_downsample_mesh_range = self.random_downsample_mesh_range

                    # res = torch.tensor(
                        # mesh.matrix_size, device=representation.device
                    # ).float()
                    # if random_downsample_mesh:
                        # random_factor = random_downsample_mesh_range[0] + torch.rand(
                            # (1,), device=representation.device
                        # ).item() * (
                            # random_downsample_mesh_range[1]
                            # - random_downsample_mesh_range[0]
                        # )
                        # res = torch.ceil(res * random_factor).int()
                    # res[volume_index] = self.slab_thickness

                    # slices = representation.forward(
                        # mesh.add_cont_slice_select(
                            # axis=volume_index,
                            # mesh_lbs=slices_lb,
                            # mesh_ubs=slices_ub,
                            # mesh_resolutions=res,
                        # )
                    # )

                    # ret_slice_inds.append(slices_lb)

                if not keep_dim:
                    slices = slices.moveaxis(volume_index, 0)

                if averageing:
                    slices = slices.view(
                        slab_thickness, rep, -1, slices.shape[-2], slices.shape[-1]
                    ).mean(dim=0)
                elif not keep_dim:
                    slices = slices.squeeze(1)

                if swapping and self.stack_num_slices == 1:
                    slices = slices.swapaxes(-2, -3)

                if self.stack_num_slices > 1:
                    if not self.slices_discrete:
                        raise NotImplementedError(
                            "stack_num_slices > 1 currently requires slices_discrete=True."
                        )
                    if keep_dim:
                        raise NotImplementedError(
                            "stack_num_slices > 1 currently requires keep_dims=False."
                        )
                    if averageing:
                        raise NotImplementedError(
                            "stack_num_slices > 1 currently requires average_slabs=False."
                        )

                    slice_dim = mesh.matrix_size[volume_index]
                    center_inds = slice_inds.int()
                    half = self.stack_num_slices // 2
                    if self.stack_num_slices % 2 == 0:
                        offsets = torch.arange(
                            -half + 1,
                            half + 1,
                            device=representation.device,
                        ).view(1, -1)
                    else:
                        offsets = torch.arange(
                            -half,
                            half + 1,
                            device=representation.device,
                        ).view(1, -1)
                    stack_inds = center_inds.unsqueeze(1) + offsets

                    if self.stack_padding_mode == "edge":
                        stack_inds = stack_inds.clamp(0, slice_dim - 1)

                    flat_stack_inds = stack_inds.reshape(-1)
                    stack_slices = representation.forward(
                        mesh.add_index_select(volume_index, flat_stack_inds.int())
                    )
                    stack_slices = stack_slices.moveaxis(volume_index, 0).squeeze(1)
                    stack_slices = stack_slices.reshape(
                        center_inds.shape[0],
                        self.stack_num_slices,
                        *stack_slices.shape[1:]
                    )

                    if stack_slices.ndim != 5:
                        raise ValueError(
                            f"Expected stacked slice tensor of ndim=5 (N, S, H, W, C), got shape {stack_slices.shape}."
                        )

                    if swapping:
                        stack_slices = stack_slices.swapaxes(-2, -3)

                    slices = stack_slices

                ret.append(slices)

        return ret, ret_slice_inds