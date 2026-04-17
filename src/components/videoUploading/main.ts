/* eslint-disable no-use-before-define */
/* eslint-disable consistent-return */
/* eslint-disable guard-for-in */
/* eslint-disable no-restricted-syntax */
/* eslint-disable no-underscore-dangle */
/* eslint-disable no-console */
/* eslint-disable no-param-reassign */
import {
  NeuralPipeline, // Interface
  // original
  Original,
  // Upscale
  GANx3L,
  CNNx2UL,
  GANx4UUL,
  // Restore
  CNNUL,
  GANUUL,
  CNN,
  NeuralPipelineDescriptor,
} from 'webgpu-neural';

import type { JSX } from 'react';

import { makeSample, SampleInit } from '../SampleLayout';
import CanvasControls from '../CanvasControls';

import fullscreenTexturedQuadWGSL from '../../shaders/fullscreenTexturedQuad.wgsl';
import sampleExternalTextureWGSL from '../../shaders/sampleExternalTexture.frag.wgsl';

type Settings = {
  requestFrame: string;
  effect: string;
  compareOn: boolean;
  splitRatio: number;
};

async function configureWebGPU(canvas: HTMLCanvasElement) {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  });

  return { device, context, presentationFormat };
}

const init: SampleInit = async ({
  canvas, pageState, gui, stats, videoURL,
}) => {
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  // Set video element
  const video = document.createElement('video');
  video.autoplay = true;
  video.muted = true;
  video.controls = true;

  video.addEventListener('loadedmetadata', () => {
    console.log(`Updating video metadata to ${video.videoHeight}p`);
  });

  let player: dashjs.MediaPlayerClass | undefined; // Declare at top-level
  if (typeof window !== 'undefined' && typeof window.self !== 'undefined') {
    // This ensures that the code only runs in the browser
    const dashjs = await import('dashjs');
    player = dashjs.MediaPlayer().create();
    player.initialize(video, videoURL, true); // videoURL is your manifest.mpd
  }

  const controlsPlaceholder = document.getElementById('canvasControls');
  const controls = new CanvasControls(video, canvas, controlsPlaceholder);

  if (controlsPlaceholder) {
    controlsPlaceholder.appendChild(controls.container);
  }

  await video.play();

  let WIDTH = video.videoWidth;
  let HEIGHT = video.videoHeight;

  if (!pageState.active) return;

  const { device, context, presentationFormat } = await configureWebGPU(canvas);

  let videoFrameTexture: GPUTexture;
  videoFrameTexture = device.createTexture({
    size: [WIDTH, HEIGHT, 1],
    format: 'rgba16float',
    usage: GPUTextureUsage.TEXTURE_BINDING
    | GPUTextureUsage.COPY_DST
    | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  function updateVideoFrameTexture() {
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.log('Found invalid frame... Skipping...');
      return; // Invalid frame, skip
    }

    if (video.videoWidth !== WIDTH || video.videoHeight !== HEIGHT) {
      console.log(`Resolution updated to: ${video.videoWidth}x${video.videoHeight}`);
      WIDTH = video.videoWidth;
      HEIGHT = video.videoHeight;
      videoFrameTexture = device.createTexture({
        size: [WIDTH, HEIGHT, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING
        | GPUTextureUsage.COPY_DST
        | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      updatePipeline();
      updateRenderBindGroup();
      updateCanvasSize();
    }

    device.queue.copyExternalImageToTexture(
      { source: video },
      { texture: videoFrameTexture },
      [video.videoWidth, video.videoHeight],
    );
  }

  // bind 2: compare
  const compareBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // bind 4: compare split ratio
  const splitRatioBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const settings: Settings = {
    requestFrame: 'requestVideoFrameCallback',
    effect: 'Original',
    compareOn: false,
    splitRatio: 50,
  };

  // initial pipline mode
  let customPipeline: NeuralPipeline;
  function updatePipeline() {
    const pipelineDescriptor: NeuralPipelineDescriptor = {
      device,
      inputTexture: videoFrameTexture,
    };
    switch (settings.effect) {
      case 'Original':
        customPipeline = new Original({ inputTexture: videoFrameTexture });
        break;
      // Upscale
      case 'Upscale-GANx3L':
        customPipeline = new GANx3L(pipelineDescriptor);
        break;
      case 'Upscale-CNNx2UL':
        customPipeline = new CNNx2UL(pipelineDescriptor);
        break;
      case 'Upscale-GANx4UUL':
        customPipeline = new GANx4UUL(pipelineDescriptor);
        break;
      // Restore
      case 'Restore-CNNUL':
        customPipeline = new CNNUL(pipelineDescriptor);
        break;
      case 'Restore-GANUUL':
        customPipeline = new GANUUL(pipelineDescriptor);
        break;
      case 'CNN':
        customPipeline = new CNN(pipelineDescriptor);
        break;
      default:
        console.log('Invalid selection');
        break;
    }
  }
  updatePipeline();

  function updateCanvasSize() {
    // setting canvas dimensions
    canvas.width = customPipeline.getOutputTexture().width;
    canvas.height = customPipeline.getOutputTexture().height;
    // canvas.style.width = `${customPipeline.getOutputTexture().width}px`;
    // canvas.style.height = `${customPipeline.getOutputTexture().height}px`;
    canvas.style.width = '1280px';
    canvas.style.height = '720px';
  }
  updateCanvasSize();
  controls.fitToCanvasSize();

  for (const folder in gui.__folders) {
    gui.removeFolder(gui.__folders[folder]);
  }
  while (gui.__controllers.length > 0) {
    gui.__controllers[0].remove();
  }
  const generalFolder = gui.addFolder('General');
  const effectController = generalFolder.add(
    settings,
    'effect',
    [
      'Original',
      // Upscale
      'Upscale-GANx3L',
      'Upscale-CNNx2UL',
      'Upscale-GANx4UUL',
      // Restore
      'Restore-CNNUL',
      'Restore-GANUUL',
      'CNN',
    ],
  )
    .name('Effect');

  function downloadCanvasAsImage() {
    const downloadLink = document.createElement('a');
    downloadLink.setAttribute('download', 'CanvasImage.png');
    console.log('download canvas of size', canvas.width, canvas.height);

    // Convert canvas content to data URL
    canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      downloadLink.setAttribute('href', url);
      downloadLink.click();
    }, 'image/png');
  }

  generalFolder.add({ downloadCanvasAsImage }, 'downloadCanvasAsImage').name('Download Canvas');

  generalFolder.add(settings, 'compareOn')
    .name('Comparison')
    .onChange((value) => {
      device.queue.writeBuffer(compareBuffer, 0, new Uint32Array([value ? 1 : 0]));
      oneFrame();
    });
  generalFolder.add(settings, 'splitRatio', 0, 100, 0.1)
    .name('Split Line%')
    .onChange((value) => {
      device.queue.writeBuffer(splitRatioBuffer, 0, new Float32Array([value / 100]));
      oneFrame();
    });

  // initial comparsion setting
  if (settings.compareOn) {
    device.queue.writeBuffer(compareBuffer, 0, new Uint32Array([1]));
  } else {
    device.queue.writeBuffer(compareBuffer, 0, new Uint32Array([0]));
  }
  device.queue.writeBuffer(splitRatioBuffer, 0, new Float32Array([settings.splitRatio / 100]));

  // configure final rendering pipeline
  const renderBindGroupLayout = device.createBindGroupLayout({
    label: 'Render Bind Group Layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {},
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {},
      },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {},
      },
      {
        binding: 4,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
    ],
  });

  const renderPipelineLayout = device.createPipelineLayout({
    label: 'Render Pipeline Layout',
    bindGroupLayouts: [renderBindGroupLayout],
  });

  const renderPipeline = device.createRenderPipeline({
    layout: renderPipelineLayout,
    vertex: {
      module: device.createShaderModule({
        code: fullscreenTexturedQuadWGSL,
      }),
      entryPoint: 'vert_main',
    },
    fragment: {
      module: device.createShaderModule({
        code: sampleExternalTextureWGSL,
      }),
      entryPoint: 'main',
      targets: [
        {
          format: presentationFormat,
        },
      ],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  // bind 0: sampler
  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  // configure render pipeline
  let renderBindGroup: GPUBindGroup;
  function updateRenderBindGroup() {
    renderBindGroup = device.createBindGroup({
      layout: renderBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: sampler,
        },
        {
          binding: 1,
          resource: customPipeline.getOutputTexture().createView(),
        },
        {
          binding: 2,
          resource: {
            buffer: compareBuffer,
          },
        },
        {
          binding: 3,
          resource: videoFrameTexture.createView(),
        },
        {
          binding: 4,
          resource: {
            buffer: splitRatioBuffer,
          },
        },
      ],
    });
  }

  updateRenderBindGroup();

  effectController.onChange(() => {
    updatePipeline();
    updateRenderBindGroup();
    updateCanvasSize();
    controls.fitToCanvasSize();
    oneFrame();
  });

  for (const folder in gui.__folders) {
    gui.__folders[folder].open();
  }

  function oneFrame() {
    if (!video.paused) {
      return;
    }
    updateVideoFrameTexture();
    // initialize command recorder
    const commandEncoder = device.createCommandEncoder();

    // encode compute pipeline commands
    customPipeline.pass(commandEncoder);

    // dispatch render pipeline
    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: {
            r: 0.0, g: 0.0, b: 0.0, a: 1.0,
          },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    passEncoder.setPipeline(renderPipeline);
    passEncoder.setBindGroup(0, renderBindGroup);
    passEncoder.draw(6);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
  }

  function frame() {
    stats.begin();
    // fetch a new frame from video element into texture
    if (!video.paused) {
      // fetch a new frame from video element into texture
      updateVideoFrameTexture();
    }

    // initialize command recorder
    const commandEncoder = device.createCommandEncoder();

    // encode compute pipeline commands
    customPipeline.pass(commandEncoder);

    // dispatch render pipeline
    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: {
            r: 0.0, g: 0.0, b: 0.0, a: 1.0,
          },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    passEncoder.setPipeline(renderPipeline);
    passEncoder.setBindGroup(0, renderBindGroup);
    passEncoder.draw(6);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    stats.end();
    video.requestVideoFrameCallback(frame);
  }

  video.requestVideoFrameCallback(frame);

  const destroy = () => {
    video.pause();
    video.load();
    for (const folder in gui.__folders) {
      gui.removeFolder(gui.__folders[folder]);
    }
    player.destroy();
    controls.destroy();
    console.log('previous loop destroyed');
  };

  return destroy;
};

const VideoUploading: () => JSX.Element = () => makeSample({
  name: 'WebGPU Accelerated Neural-Enhanced Streaming',
  description: '',
  gui: true,
  init,
  filename: __filename,
});

export default VideoUploading;
