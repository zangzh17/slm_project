# nas_mapper.py
"""
NAS网络驱动器映射工具
用于自动映射网络共享文件夹到本地驱动器号
"""

import subprocess
import os
from typing import Optional, Tuple


class NASMapper:
    """NAS驱动器映射类"""
    
    def __init__(self, 
                 drive_letter: str = 'Z:',
                 network_path: str = r'\\iOptics-NIR-II\data',
                 username: str = 'iOptics',
                 password: str = 'iOpticsLab'):
        """
        初始化NAS映射器
        
        参数:
            drive_letter: 要映射的驱动器号 (如 'Z:')
            network_path: 网络共享路径
            username: NAS用户名
            password: NAS密码
        """
        self.drive_letter = drive_letter
        self.network_path = network_path
        self.username = username
        self.password = password
    
    def map_drive(self, persistent: bool = False) -> Tuple[bool, str]:
        """
        映射网络驱动器
        
        参数:
            persistent: 是否持久化映射（重启后仍保留）
        
        返回:
            (成功标志, 消息)
        """
        cmd = ['net', 'use', self.drive_letter, self.network_path, 
               f'/user:{self.username}', self.password]
        
        if persistent:
            cmd.append('/persistent:yes')
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return True, f"✓ 映射成功: {self.drive_letter} -> {self.network_path}"
            else:
                return False, f"✗ 映射失败: {result.stderr}"
        
        except subprocess.TimeoutExpired:
            return False, "✗ 映射超时"
        except Exception as e:
            return False, f"✗ 发生错误: {str(e)}"
    
    def unmap_drive(self) -> Tuple[bool, str]:
        """
        取消映射网络驱动器
        
        返回:
            (成功标志, 消息)
        """
        try:
            result = subprocess.run(
                ['net', 'use', self.drive_letter, '/delete', '/yes'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                return True, f"✓ 已取消映射: {self.drive_letter}"
            else:
                return False, f"✗ 取消映射失败: {result.stderr}"
        
        except Exception as e:
            return False, f"✗ 发生错误: {str(e)}"
    
    def check_mapping(self) -> bool:
        """
        检查驱动器是否已映射并可访问
        
        返回:
            是否可访问
        """
        return os.path.exists(self.drive_letter)
    
    def get_path(self, relative_path: str = '') -> str:
        """
        获取映射驱动器上的完整路径
        
        参数:
            relative_path: 相对路径
        
        返回:
            完整路径
        """
        if relative_path:
            return os.path.join(self.drive_letter, relative_path)
        return self.drive_letter
    
    def ensure_mapped(self, persistent: bool = False) -> Tuple[bool, str]:
        """
        确保驱动器已映射（如果未映射则自动映射）
        
        参数:
            persistent: 是否持久化映射
        
        返回:
            (成功标志, 消息)
        """
        if self.check_mapping():
            return True, f"✓ 驱动器已映射: {self.drive_letter}"
        else:
            return self.map_drive(persistent=persistent)


# 便捷函数
def quick_map(drive_letter: str = 'Z:',
              network_path: str = r'\\iOptics-NIR-II\data',
              username: str = 'iOptics',
              password: str = 'iOpticsLab',
              persistent: bool = False) -> Tuple[bool, str]:
    """
    快速映射NAS驱动器的便捷函数
    
    返回:
        (成功标志, 消息)
    """
    mapper = NASMapper(drive_letter, network_path, username, password)
    return mapper.ensure_mapped(persistent=persistent)


if __name__ == "__main__":
    # 命令行测试
    mapper = NASMapper()
    success, message = mapper.ensure_mapped()
    print(message)
    
    if success:
        save_dir = mapper.get_path(r'SLM_super_resolution\data')
        print(f"数据保存路径: {save_dir}")
        print(f"路径是否存在: {os.path.exists(save_dir)}")