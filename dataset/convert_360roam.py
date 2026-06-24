import os
import json
import shutil
import cv2
import argparse
from pathlib import Path
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Convert OpenMVG dataset to Split-OpenMVG format (Train + Test).")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to the original dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the target dataset directory")
    return parser.parse_args()

def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    # 1. 创建目标目录结构
    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    # 2. 定位源文件 (同时获取 train 和 test 的 JSON)
    poses_json_path = source_dir / "openMVG" / "reconstruction" / "poses.json"
    test_json_path = source_dir / "openMVG" / "data_openmvg_test.json"
    source_images_dir = source_dir / "images"
    source_ply_path = source_dir / "scene.ply"

    if not poses_json_path.exists():
        raise FileNotFoundError(f"Cannot find train poses file: {poses_json_path}")

    # 3. 解析训练集 (Train) 的 poses.json
    print(f"Reading Train Data: {poses_json_path}...")
    with open(poses_json_path, 'r') as f:
        openmvg_data = json.load(f)

    train_views = openmvg_data.get("views", [])
    train_extrinsics = openmvg_data.get("extrinsics", [])

    # 4. 解析测试集 (Test) 并动态处理 ID 偏移拼接
    test_views = []
    test_extrinsics = []
    
    if test_json_path.exists():
        print(f"Reading Test Data: {test_json_path}...")
        with open(test_json_path, 'r') as f:
            test_data = json.load(f)
        
        raw_test_views = test_data.get("views", [])
        raw_test_extrinsics = test_data.get("extrinsics", [])
        
        # 寻找训练集中的最大 ID，用于为测试集生成绝对不冲突的新 ID (比如顺延到 75)
        max_key = -1
        max_ptr_id = 2147483648
        for v in train_views:
            max_key = max(max_key, v.get("key", -1))
            max_key = max(max_key, v["value"]["ptr_wrapper"]["data"].get("id_pose", -1))
            max_ptr_id = max(max_ptr_id, v["value"]["ptr_wrapper"].get("id", max_ptr_id))
        for e in train_extrinsics:
            max_key = max(max_key, e.get("key", -1))
            
        offset = max_key + 1
        ptr_offset = max_ptr_id + 1
        
        # 映射并顺延 Test 的 Extrinsics
        test_ext_map = {}
        for e in raw_test_extrinsics:
            old_e_key = e["key"]
            new_e_key = old_e_key + offset
            test_ext_map[old_e_key] = new_e_key
            
            e_copy = copy.deepcopy(e)
            e_copy["key"] = new_e_key
            test_extrinsics.append(e_copy)
            
        # 映射并顺延 Test 的 Views
        for i, v in enumerate(raw_test_views):
            v_copy = copy.deepcopy(v)
            old_key = v_copy["key"]
            new_key = old_key + offset
            old_id_pose = v_copy["value"]["ptr_wrapper"]["data"]["id_pose"]
            
            v_copy["key"] = new_key
            v_copy["value"]["ptr_wrapper"]["id"] = ptr_offset + i
            v_copy["value"]["ptr_wrapper"]["data"]["id_view"] = new_key
            v_copy["value"]["ptr_wrapper"]["data"]["id_pose"] = test_ext_map.get(old_id_pose, new_key)
            
            test_views.append(v_copy)
    else:
        print(f"Warning: Test file {test_json_path} not found. Proceeding with train data only.")

    # 5. 合并全部相机内外参数据
    all_views = train_views + test_views
    all_extrinsics = train_extrinsics + test_extrinsics

    sfm_data_version = openmvg_data.get("sfm_data_version", "0.3")
    root_path = str(out_images_dir.resolve()) + "/"

    # 构造并剥离外参 (Extrinsics) 字典
    data_extrinsics = {
        "sfm_data_version": sfm_data_version,
        "root_path": root_path,
        "views": [],
        "intrinsics": [],
        "extrinsics": all_extrinsics,
        "structure": [],
        "control_points": []
    }

    # 构造并处理视角 (Views) 字典，包含统一降采样
    print(f"Processing {len(all_views)} total images (downsampling to 2048x1024)...")
    updated_views = []
    
    for view in all_views:
        view_data = copy.deepcopy(view)
        ptr_data = view_data["value"]["ptr_wrapper"]["data"]
        filename = ptr_data["filename"]
        
        src_img_path = source_images_dir / filename
        dst_img_path = out_images_dir / filename
        
        # 对图片进行尺寸校验与等比例缩放降采样 (INTER_AREA 对图像质量最好)
        if src_img_path.exists():
            img = cv2.imread(str(src_img_path))
            if img is not None:
                img_resized = cv2.resize(img, (2048, 1024), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst_img_path), img_resized)
            else:
                print(f"Warning: Failed to read {src_img_path}")
        else:
            print(f"Warning: Image not found {src_img_path}")

        # 将 JSON 里的分辨率重置为 2048x1024，与物理图片对应
        ptr_data["width"] = 2048
        ptr_data["height"] = 1024
        
        updated_views.append(view_data)

    data_views = {
        "sfm_data_version": sfm_data_version,
        "root_path": root_path,
        "views": updated_views,
        "intrinsics": [],
        "extrinsics": [],
        "structure": [],
        "control_points": []
    }

    # 6. 将合并完的数据持久化到磁盘
    with open(output_dir / "data_extrinsics.json", 'w') as f:
        json.dump(data_extrinsics, f, indent=4)
        
    with open(output_dir / "data_views.json", 'w') as f:
        json.dump(data_views, f, indent=4)

    # 7. 复制稀疏点云
    if source_ply_path.exists():
        print(f"Copying point cloud from {source_ply_path} to pcd.ply...")
        shutil.copy(source_ply_path, output_dir / "pcd.ply")
    else:
        alt_ply_path = source_dir / "openMVG" / "reconstruction" / "colorized.ply"
        if alt_ply_path.exists():
            print(f"Copying colorized.ply to pcd.ply...")
            shutil.copy(alt_ply_path, output_dir / "pcd.ply")
        else:
            print("Warning: Could not find point cloud (scene.ply or colorized.ply) to copy.")

    # 8. 依据原始划分提取生成 train.txt 与 test.txt
    print("Generating train/test split from pose_c2w.json...")
    pose_c2w_path = source_dir / "pose_c2w.json"
    
    if pose_c2w_path.exists():
        with open(pose_c2w_path, 'r') as f:
            pose_c2w_data = json.load(f)
            
        train_list = []
        if "train" in pose_c2w_data:
            for frame in pose_c2w_data["train"]:
                base_name = os.path.splitext(frame["rgb_file"])[0]
                train_list.append(base_name)
                
        test_list = []
        if "test" in pose_c2w_data:
            for frame in pose_c2w_data["test"]:
                base_name = os.path.splitext(frame["rgb_file"])[0]
                test_list.append(base_name)
                
        with open(output_dir / "train.txt", 'w') as f:
            for name in train_list:
                f.write(f"{name}\n")
                
        with open(output_dir / "test.txt", 'w') as f:
            for name in test_list:
                f.write(f"{name}\n")
                
        print(f"Split complete! Train: {len(train_list)} images, Test: {len(test_list)} images.")
    else:
        print(f"Error: {pose_c2w_path} not found! Cannot generate train/test splits.")

    print(f"Conversion complete! Target dataset saved at: {output_dir}")

if __name__ == "__main__":
    main()

# python convert_dataset.py \
#     --source_dir /data1/xuyihang/pano/dataset/360Roam_ori/cafe \
#     --output_dir /data1/xuyihang/pano/dataset/360Roam_new/cafe 