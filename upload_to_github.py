#!/usr/bin/env python3
"""
GitHub 文件上传脚本
使用 GitHub API 上传文件到仓库
"""

import os
import sys
import base64
import json
import urllib.request
import urllib.error

# 配置
REPO_OWNER = "Rusiansda"
REPO_NAME = "major-bleeding-prediction"
BRANCH = "main"

# 需要上传的文件
FILES_TO_UPLOAD = [
    "model_CatBoost_calibrated.pkl",
    "robust_scaler.pkl",
    "utils.py",
    "major_bleeding_web_deploy.py",
    "requirements.txt",
    "README.md",
    ".gitignore"
]

def get_file_content(filepath):
    """读取文件内容并进行 base64 编码"""
    with open(filepath, 'rb') as f:
        content = f.read()
    return base64.b64encode(content).decode('utf-8')

def upload_file_to_github(token, filepath, commit_message="Update files"):
    """上传单个文件到 GitHub"""
    filename = os.path.basename(filepath)
    
    # API URL
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{filename}"
    
    # 首先检查文件是否存在（获取 sha）
    sha = None
    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", f"token {token}")
        req.add_header("Accept", "application/vnd.github.v3+json")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                sha = data.get('sha')
                print(f"  文件 {filename} 存在，将更新 (sha: {sha[:8]}...)")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  文件 {filename} 不存在，将创建新文件")
        else:
            print(f"  检查文件状态时出错: {e}")
    except Exception as e:
        print(f"  检查文件状态时出错: {e}")
    
    # 准备请求数据
    content = get_file_content(filepath)
    data = {
        "message": commit_message,
        "content": content,
        "branch": BRANCH
    }
    if sha:
        data["sha"] = sha
    
    # 发送 PUT 请求
    req = urllib.request.Request(url, method="PUT")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("Content-Type", "application/json")
    
    json_data = json.dumps(data).encode('utf-8')
    req.add_header("Content-Length", str(len(json_data)))
    
    try:
        with urllib.request.urlopen(req, json_data, timeout=60) as response:
            if response.status in [200, 201]:
                print(f"✅ {filename} 上传成功")
                return True
            else:
                print(f"❌ {filename} 上传失败: HTTP {response.status}")
                return False
    except urllib.error.HTTPError as e:
        print(f"❌ {filename} 上传失败: {e.code} - {e.read().decode('utf-8')}")
        return False
    except Exception as e:
        print(f"❌ {filename} 上传失败: {str(e)}")
        return False

def main():
    print("="*60)
    print("GitHub 文件上传工具")
    print("="*60)
    print(f"目标仓库: {REPO_OWNER}/{REPO_NAME}")
    print(f"分支: {BRANCH}")
    print("="*60)
    
    # 从环境变量或用户输入获取 token
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        token = input("\n请输入 GitHub Personal Access Token: ").strip()
    
    if not token:
        print("❌ 错误: 需要提供 GitHub Token")
        print("\n如何获取 Token:")
        print("1. 访问 https://github.com/settings/tokens")
        print("2. 点击 'Generate new token (classic)'")
        print("3. 勾选 'repo' 权限")
        print("4. 生成并复制 token")
        sys.exit(1)
    
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"\n工作目录: {os.getcwd()}")
    print(f"准备上传 {len(FILES_TO_UPLOAD)} 个文件...\n")
    
    success_count = 0
    for filepath in FILES_TO_UPLOAD:
        if not os.path.exists(filepath):
            print(f"⚠️ 跳过不存在的文件: {filepath}")
            continue
        
        print(f"上传 {filepath}...")
        if upload_file_to_github(token, filepath, "Update: Best CatBoost model (AUC 0.8337)"):
            success_count += 1
        print()
    
    print("="*60)
    print(f"上传完成: {success_count}/{len(FILES_TO_UPLOAD)} 个文件成功")
    print("="*60)
    
    if success_count == len(FILES_TO_UPLOAD):
        print("\n✅ 所有文件上传成功!")
        print(f"访问: https://github.com/{REPO_OWNER}/{REPO_NAME}")
    else:
        print("\n⚠️ 部分文件上传失败，请检查错误信息")

if __name__ == "__main__":
    main()
