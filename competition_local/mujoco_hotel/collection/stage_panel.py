"""Chinese collection status in a companion window; never sends robot commands."""
import argparse,json,os,tkinter as tk
from pathlib import Path
from .schema import STAGES


def main():
    p=argparse.ArgumentParser();p.add_argument('--status',type=Path,required=True)
    p.add_argument('--parent-pid',type=int);args=p.parse_args()
    root=tk.Tk();root.title('MuJoCo 采集阶段');root.geometry('360x195+16+42');root.attributes('-topmost',True)
    root.configure(bg='#18212b');text=tk.StringVar(value='等待采集窗口启动')
    panel=tk.Label(root,textvariable=text,font=('Microsoft YaHei UI',10),fg='white',bg='#18212b',
             justify='left',anchor='nw',wraplength=336,padx=10,pady=8)
    panel.pack(fill='both',expand=True)
    root.bind('<Configure>',lambda event:panel.configure(wraplength=max(120,root.winfo_width()-24)))
    parent_handle=None
    if args.parent_pid and os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        parent_handle=kernel.OpenProcess(0x00100000,False,args.parent_pid)
    def poll():
        if parent_handle and kernel.WaitForSingleObject(parent_handle,0)==0:
            root.destroy();return
        try:
            d=json.loads(args.status.read_text(encoding='utf-8'));stage=int(d['stage'])
            mode={'walk':'1 行走','stair':'2 楼梯','low_grasp':'4 低位抓取','high_grasp':'5 高位抓取'}.get(d.get('mode'),'等待同步')
            message=d.get('message','')
            extra='\n'+message if d.get('error') or message.startswith('Saved:') else ''
            text.set(f"{'录制中' if d.get('recording') else '未录制'} · 阶段 {stage}/14　|　{mode}\n\n"
                +STAGES[stage-1]['instruction_zh']+'\n\nF1 开始　F2 下一阶段　F3 保存\nF6 中断保留 partial'+extra)
        except (OSError,ValueError,KeyError,IndexError):pass
        root.after(250,poll)
    root.after(250,poll)
    try:root.mainloop()
    finally:
        if parent_handle:kernel.CloseHandle(parent_handle)


if __name__=='__main__':main()
