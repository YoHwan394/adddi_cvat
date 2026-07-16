# tinySmoke_260710 CVAT serverless function

This Nuclio function exposes `best.pt` to CVAT as a rectangle detector with one label: `smoke`.

## Deploy

From the CVAT repository root:

```bash
./serverless/deploy_cpu.sh serverless/tinySmoke_260710
```

The function is shown in CVAT as `tinySmoke_260710`. The CVAT auto-annotation confidence value is forwarded to Ultralytics as the detection threshold. If CVAT does not send a value, the default is `0.25`.
