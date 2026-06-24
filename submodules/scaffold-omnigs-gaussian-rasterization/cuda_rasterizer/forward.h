/**
 * This file is part of OmniGS
 *
 * Copyright (C) 2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * OmniGS is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * OmniGS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with OmniGS.
 * If not, see <http://www.gnu.org/licenses/>.
 * 
 * OmniGS is Derivative Works of Gaussian Splatting.
 * Its usage should not break the terms in LICENSE.md.
 */

 #ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
 #define CUDA_RASTERIZER_FORWARD_H_INCLUDED
 
 #include <cuda.h>
 #include "cuda_runtime.h"
 #include "device_launch_parameters.h"
 #define GLM_FORCE_CUDA
 #include <glm/glm.hpp>
 
 namespace FORWARD
 {
	 // Perform initial steps for each Gaussian prior to rasterization.
	 void preprocess(int P, int D, int M,
		 const float* orig_points,
		 const glm::vec3* scales,
		 const float scale_modifier,
		 const glm::vec4* rotations,
		 const float* opacities,
		 const float* shs,
		 bool* clamped,
		 const float* cov3D_precomp,
		 const float* colors_precomp,
		 const float* viewmatrix,
		 const float* projmatrix,
		 const glm::vec3* cam_pos,
		 const int W, int H,
		 const float focal_x, float focal_y,
		 const float tan_fovx, float tan_fovy,
		 int* radii,
		 float2* points_xy_image,
		 float* depths,
		 float* cov3Ds,
		 float* colors,
		 float4* conic_opacity,
		 const dim3 grid,
		 uint32_t* tiles_touched,
		 bool prefiltered);
 
	 // Main rasterization method.
	 void render(
		 const dim3 grid, dim3 block,
		 const uint2* ranges,
		 const uint32_t* point_list,
		 int W, int H,
		 const float2* points_xy_image,
		 const float* features,
		 const float4* conic_opacity,
		 float* final_T,
		 uint32_t* n_contrib,
		 const float* bg_color,
		 float* out_color);
 
	 // Main rasterization method.
	 void renderDepth(
		 const dim3 grid, dim3 block,
		 const uint2* ranges,
		 const uint32_t* point_list,
		 int W, int H,
		 const float2* points_xy_image,
		 // const float* orig_points,
		 // const float* viewmatrix,
		 const float* depths,
		 const float4* conic_opacity,
		 float* final_T,
		 uint32_t* n_contrib,
		 const float* bg_color,
		 float* out_color);
 
	 // Perform initial steps for each Gaussian prior to rasterization.
	 void preprocessLonlat(int P, int D, int M,
		 const float* orig_points,
		 const glm::vec3* scales,
		 const float scale_modifier,
		 const glm::vec4* rotations,
		 const float* opacities,
		 const float* shs,
		 bool* clamped,
		 const float* cov3D_precomp,
		 const float* colors_precomp,
		 const float* viewmatrix,
		 const glm::vec3* cam_pos,
		 const int W, int H,
		 int* radii,
		 float* lat,
		 float* lon,
 // int* radii_x,
		 float2* points_xy_image,
		 float* depths,
		 float* cov3Ds,
		 float* colors,
		 float4* conic_opacity,
		 const dim3 grid,
		 uint32_t* tiles_touched,
		 bool prefiltered);
 
	 // // Main rasterization method.
	 // void renderLonlat(
	 // 	const dim3 grid, dim3 block,
	 // 	const uint2* ranges,
	 // 	const uint32_t* point_list,
	 // 	int W, int H,
	 // 	const float2* points_xy_image,
	 // 	const float* features,
	 // 	const float4* conic_opacity,
	 // 	float* final_T,
	 // 	uint32_t* n_contrib,
	 // 	const float* bg_color,
	 // 	float* out_color);

	 void filter_preprocess(int P, int M,
		const float* means3D,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* cov3D_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float* cov3Ds,
		const dim3 grid,
		bool prefiltered);

 }
 
 
 #endif