// images-generate.js
import fs from "node:fs";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const prompt =
  "Neon cosmic tarot altar, blacklight style, sharp details, poster-ready";

const res = await client.images.generate({
  model: "gpt-image-1",
  prompt,
  size: "1024x1024", // 256x256, 512x512, 1024x1024
  n: 1, // number of images
});

// The API returns base64-encoded image(s)
const b64 = res.data[0].b64_json;
fs.writeFileSync("output.png", Buffer.from(b64, "base64"));
console.log("Saved to output.png");
