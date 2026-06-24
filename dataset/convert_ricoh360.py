import os
import json
import shutil
import argparse
from pathlib import Path
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Ricoh360 dataset to Split-OpenMVG format (Train + Test).")
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

    # 2. 定位源文件 (根据 Ricoh360 的目录结构)
    train_json_path = source_dir / "openMVG" / "data_openmvg_train.json"
    test_json_path = source_dir / "openMVG" / "data_openmvg_test.json"
    source_images_dir = source_dir / "imgs"  # 注意：这里是 imgs 而不是 images
    source_ply_path = source_dir / "openMVG" / "scene.ply"
    
    source_train_txt = source_dir / "train.txt"
    source_test_txt = source_dir / "test.txt"

    if not train_json_path.exists():
        raise FileNotFoundError(f"Cannot find train poses file: {train_json_path}")

    # 3. 解析训练集 (Train) 的 JSON
    print(f"Reading Train Data: {train_json_path}...")
    with open(train_json_path, 'r') as f:
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
        
        # 寻找训练集中的最大 ID，用于为测试集生成绝对不冲突的新 ID
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

    # 构造并处理视角 (Views) 字典，直接复制原图不进行降采样
    print(f"Processing {len(all_views)} total images (Copying original images directly)...")
    updated_views = []
    
    for view in all_views:
        view_data = copy.deepcopy(view)
        ptr_data = view_data["value"]["ptr_wrapper"]["data"]
        filename = ptr_data["filename"]
        
        src_img_path = source_images_dir / filename
        dst_img_path = out_images_dir / filename
        
        # 检查图片是否存在并进行纯物理拷贝
        if src_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image not found {src_img_path}")

        # 注意：这里我们不再修改 ptr_data["width"] 和 ptr_data["height"]，完全保留原 JSON 数值
        
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
        shutil.copy2(source_ply_path, output_dir / "pcd.ply")
    else:
        print(f"Warning: Could not find point cloud ({source_ply_path}) to copy.")

    # 8. 直接复制原有的 train.txt 和 test.txt
    print("Copying train.txt and test.txt...")
    if source_train_txt.exists():
        shutil.copy2(source_train_txt, output_dir / "train.txt")
        print("  - train.txt copied.")
    else:
        print(f"Warning: {source_train_txt} not found!")

    if source_test_txt.exists():
        shutil.copy2(source_test_txt, output_dir / "test.txt")
        print("  - test.txt copied.")
    else:
        print(f"Warning: {source_test_txt} not found!")

    print(f"🎉 Conversion complete! Target dataset saved at: {output_dir}")

if __name__ == "__main__":
    main()