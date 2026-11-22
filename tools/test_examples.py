"""Test PNG examples: fresh in-memory workflow and independent saved-forgery recovery."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

os.environ.setdefault('HF_HUB_OFFLINE','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
import numpy as np
from PIL import Image, ImageDraw
from verify import VerifyPipeline, blend_forgery
from verify.image_io import image_to_pil,load_image,load_mask,mask_to_pil,save_image
from verify.workflow import psnr,roi_psnr


def bit_grid(bits):
    values=bits.detach().cpu().numpy().astype(np.uint8).reshape(10,10)*255
    return Image.fromarray(values).resize((256,256),Image.Resampling.NEAREST)


def overlay(image,mask):
    rgb=np.asarray(image_to_pil(image)).astype(np.float32)
    alpha=mask.detach().cpu().numpy()[0,0,:,:,None]*.48
    return Image.fromarray((rgb*(1-alpha)+np.array([255,30,30])*alpha).clip(0,255).astype(np.uint8))


def panel(items,path,columns=4):
    canvas=Image.new('RGB',(276*columns,296*math.ceil(len(items)/columns)),'#e9edf2')
    draw=ImageDraw.Draw(canvas)
    for index,(label,picture) in enumerate(items):
        x,y=(index%columns)*276+10,(index//columns)*296+8
        draw.text((x,y),label,fill='#172235')
        canvas.paste(picture.convert('RGB').resize((256,256)),(x,y+28))
    canvas.save(path)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples-dir',type=Path,default=ROOT/'examples')
    parser.add_argument('--output-dir',type=Path,help='new output directory; default: unique timestamped directory in logs/')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--seed',type=int,default=2025)
    args=parser.parse_args()
    if args.output_dir is None:
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        args.output_dir=ROOT/'logs'/f'top10_retest_{stamp}'
    # Explicit destinations remain protected from accidental overwrites.
    if args.output_dir.exists():
        parser.error(f'Output directory already exists: {args.output_dir}. Omit --output-dir for a new timestamped run or choose a new path; existing results are preserved.')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    print(f'Output directory: {args.output_dir.resolve()}',flush=True)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    manifest=json.loads((args.examples_dir/'manifest.json').read_text(encoding='utf-8'))
    pipe=VerifyPipeline(device=args.device)
    rows=[];summary=[]
    for sample in manifest['examples']:
        folder=args.examples_dir/sample['directory']
        x=load_image(folder/'original.png',device=pipe.device)
        source=load_image(folder/'source.png',device=pipe.device)
        saved_forged=load_image(folder/'forged.png',device=pipe.device)
        mask=load_mask(folder/'mask.png').to(pipe.device)
        expected=json.loads((folder/'copyright.json').read_text())['expected']
        bits=torch.tensor([int(c) for c in expected],device=pipe.device)
        destination=args.output_dir/sample['directory'];destination.mkdir()
        # Generate the carrier here; it is an output, never an example input.
        carrier=pipe.embed(x,bits.tolist(),return_tensor=True)
        seed=args.seed+sample['validation_index']
        attack=blend_forgery(carrier,source,mask=mask,seed=seed,method='random')
        row=dict(sample=sample['directory'],historical_float_recovery_psnr=sample['recovery_psnr'],cases={})
        for case,forged in [('full_pipeline',attack['image']),('saved_forgery_png',saved_forged)]:
            target=destination/case;target.mkdir()
            result=pipe.recover(forged,return_tensor=True)
            copyright=pipe.extract(forged)
            predicted=pipe.locate(forged)
            accuracy=float((copyright['watermark'][0]==bits).float().mean())
            checks=dict(finite=bool(torch.isfinite(result['image']).all()),
                copyright_consistent=bool(torch.equal(copyright['watermark'],result['watermark'])),
                localization_consistent=bool(torch.equal(predicted,result['mask'])),
                complete_weights=bool(pipe.checkpoint_load_report['complete_model_load']))
            metrics=dict(recovery_psnr=psnr(result['image'],x),forged_psnr=psnr(forged,x),
                copyright_accuracy=accuracy,roi_recovery_psnr=roi_psnr(result['image'],x,mask),
                roi_baseline_psnr=roi_psnr(forged,x,mask),functional_checks=checks)
            metrics['roi_gain_db']=metrics['roi_recovery_psnr']-metrics['roi_baseline_psnr']
            if case=='full_pipeline':metrics['attack']=attack['metadata']
            else:metrics['difference_from_historical_float_psnr']=metrics['recovery_psnr']-sample['recovery_psnr']
            metrics['pass']=all(checks.values())
            for name,value in [('original',x),('source',source),('forged',forged),('recovered',result['image'])]:
                save_image(value,target/f'{name}.png')
            if case=='full_pipeline':save_image(carrier,target/'watermarked.png')
            mask_to_pil(mask,binary=True).save(target/'mask.png')
            mask_to_pil(predicted).save(target/'predicted_mask.png')
            mask_to_pil(predicted,binary=True).save(target/'predicted_mask_binary.png')
            overlay(forged,predicted).save(target/'localization_overlay.png')
            extracted=''.join(str(int(v)) for v in copyright['watermark'][0].tolist())
            (target/'copyright_extracted.txt').write_text(extracted+'\n',encoding='utf-8')
            (target/'copyright.json').write_text(json.dumps(dict(expected=expected,extracted=extracted,
                accuracy=accuracy,probabilities=copyright['confidence'][0].cpu().tolist()),indent=2),encoding='utf-8')
            bit_grid(copyright['watermark']).save(target/'copyright_extracted.png')
            items=[('Original',image_to_pil(x)),('Source (donor)',image_to_pil(source))]
            if case=='full_pipeline':items.append(('Watermarked',image_to_pil(carrier)))
            items += [('Forged',image_to_pil(forged)),('True mask',mask_to_pil(mask)),
                      ('Predicted mask',mask_to_pil(predicted)),('Localization',overlay(forged,predicted)),
                      ('Recovered',image_to_pil(result['image']))]
            panel(items,target/'comparison.png')
            (target/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
            row['cases'][case]=metrics
            if case=='full_pipeline':
                for label,picture in [('Original',image_to_pil(x)),('Forged',image_to_pil(forged)),
                                      ('Mask',mask_to_pil(mask)),('Recovered',image_to_pil(result['image']))]:
                    summary.append((sample['directory']+' '+label,picture))
        row['pass']=all(case['pass'] for case in row['cases'].values())
        (destination/'metrics.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        rows.append(row);print(json.dumps(row),flush=True)
    panel(summary,args.output_dir/'summary.png')
    report=dict(status='pass' if all(row['pass'] for row in rows) else 'fail',samples=len(rows),device=str(pipe.device),
                examples_dir=str(args.examples_dir.resolve()),input_format='PNG + JSON',output_format='PNG + JSON/TXT',
                full_pipeline_intermediate_png_reload=False,saved_forgery_png_loaded_as_input=True,
                status_scope='functional consistency only, not a PSNR or copyright quality threshold',results=rows)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# PNG 测试样例验证','','只使用 PNG 图片及 JSON/TXT 标签，不读取或生成 PT/NPZ 数据。','',
           'full_pipeline：读取原图/供体/mask 后，重新嵌入与训练兼容 blending，在内存完成提取、定位、恢复；中间 PNG 只导出，不读回。',
           'saved_forgery_png：独立读取样例 forged.png 验证已保存图片的推理。该输入含 PNG 量化，不等同原浮点验证。',
           '真实 mask 只用于攻击和评估，不交给恢复器。pass 仅检查功能一致性；不要求与历史浮点 PSNR 相等，质量指标如实报告。','',
           '[完整流程总览](summary.png) · [报告](report.json)','','| 样例 | 新完整流程 PSNR / 版权准确率 | 已保存 PNG PSNR / 版权准确率 |','| --- | --- | --- |']
    for row in rows:
        fresh,saved=[row['cases'][key] for key in ('full_pipeline','saved_forgery_png')]
        lines.append(f"| [{row['sample']}]({row['sample']}/full_pipeline/comparison.png) | {fresh['recovery_psnr']:.2f} / {fresh['copyright_accuracy']:.0%} | {saved['recovery_psnr']:.2f} / {saved['copyright_accuracy']:.0%} |")
    (args.output_dir/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'{report["status"].upper()}: {len(rows)} examples; report: {args.output_dir.resolve()/"report.json"}',flush=True)
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
