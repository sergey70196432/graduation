export const YOLO = {
  inputSize: 320,
  confidenceThreshold: 0.25,
  // Ограничение на количество боксов до NMS (чтобы не гонять огромные массивы).
  // Для YOLOv8n на 320 обычно этого хватает с запасом.
  preNmsTopK: 200,
  // После NMS тоже ограничиваем, чтобы не перегружать UI.
  postNmsTopK: 50,
} as const;

