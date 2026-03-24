import { Image } from 'react-native';

export async function loadLabelsFromAsset(
  assetModuleId: number
): Promise<string[]> {
  try {
    const source = Image.resolveAssetSource(assetModuleId);
    const uri = source?.uri;
    if (!uri) {
      console.warn('[labels] Не удалось получить uri для labels.txt');
      return [];
    }

    // В RN ассеты можно прочитать через fetch по uri.
    const text = await (await fetch(uri)).text();
    const lines = text
      .split(/\r?\n/g)
      .map(l => l.trim())
      .filter(l => l.length > 0);

    return lines;
  } catch (e) {
    console.warn('[labels] Ошибка чтения labels.txt', e);
    return [];
  }
}

