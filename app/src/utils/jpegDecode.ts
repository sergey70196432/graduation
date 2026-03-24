import * as RNFS from 'react-native-fs';
import { Buffer } from 'buffer';
import * as jpeg from 'jpeg-js';
import { AlphaType, ColorType, Skia } from '@shopify/react-native-skia';

export type RgbaImage = {
  width: number;
  height: number;
  // RGBA (4 bytes per pixel)
  data: Uint8Array;
};

function stripFileScheme(p: string) {
  return p.startsWith('file://') ? p.slice('file://'.length) : p;
}

export async function decodeJpegToRgba(pathOrUri: string): Promise<RgbaImage> {
  const uri = pathOrUri.startsWith('file://') ? pathOrUri : `file://${pathOrUri}`;

  // Быстрый путь: fetch(file://...).arrayBuffer() — обычно быстрее, чем RNFS.readFile(..., 'base64')
  // и не создаёт огромную base64-строку в памяти.
  let bytes: Uint8Array;
  const startFetch = Date.now();
  try {
    const res = await fetch(uri);
    const ab = await res.arrayBuffer();
    bytes = new Uint8Array(ab);
    console.log('[decodeJpegToRgba] fetch success');
  } catch {
    // Fallback: если fetch(file://) не сработал на конкретном девайсе/версии RN.
    const filePath = stripFileScheme(uri);
    const base64 = await RNFS.readFile(filePath, 'base64');
    bytes = new Uint8Array(Buffer.from(base64, 'base64'));
    console.log('[decodeJpegToRgba] readFile success');
  }
  console.log('[decodeJpegToRgba] fetch time', Date.now() - startFetch + 'ms');
  console.log('[decodeJpegToRgba] encoded bytes', bytes.byteLength);

  // Основной (быстрый) путь: нативный декод через Skia.
  // `jpeg-js` — чисто JS и на девайсе может занимать секунды.
  const startSkiaDecode = Date.now();
  try {
    const data = Skia.Data.fromBytes(bytes);
    const img = Skia.Image.MakeImageFromEncoded(data);
    if (!img) throw new Error('Skia.Image.MakeImageFromEncoded returned null');

    const width = img.width();
    const height = img.height();
    const rgba = img.readPixels(0, 0, {
      width,
      height,
      colorType: ColorType.RGBA_8888,
      alphaType: AlphaType.Opaque,
    });
    if (!rgba) throw new Error('Skia image.readPixels() returned null');
    if (!(rgba instanceof Uint8Array)) {
      throw new Error('Skia image.readPixels() returned Float32Array, expected Uint8Array');
    }

    console.log('[decodeJpegToRgba] skia decode+readPixels time', Date.now() - startSkiaDecode + 'ms');
    return { width, height, data: rgba };
  } catch (e) {
    console.log('[decodeJpegToRgba] skia decode failed, fallback to jpeg-js:', String(e));
  }

  // Fallback (медленно): чисто JS декодер.
  const startJsDecode = Date.now();
  const decoded = jpeg.decode(Buffer.from(bytes), { useTArray: true }) as {
    width: number;
    height: number;
    data: Uint8Array;
  };
  console.log('[decodeJpegToRgba] jpeg-js decode time', Date.now() - startJsDecode + 'ms');

  return { width: decoded.width, height: decoded.height, data: decoded.data };
}

