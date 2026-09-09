/**
 * aiVision.js
 * ============
 * DeepSeek AI Vision integration for Factory Floor Plan security analysis.
 * Analyzes floor plans for autonomous drone navigation corridors, obstacle zones,
 * and GPS spoofing reflection hazards.
 */

export const DEFAULT_DEEPSEEK_API_KEY = import.meta.env.VITE_DEEPSEEK_API_KEY || "";
const STORAGE_KEY = "sensorsentry_deepseek_key";

export function getDeepSeekApiKey() {
  return localStorage.getItem(STORAGE_KEY) || DEFAULT_DEEPSEEK_API_KEY;
}

export function setDeepSeekApiKey(key) {
  if (key && key.trim()) {
    localStorage.setItem(STORAGE_KEY, key.trim());
  } else {
    localStorage.removeItem(STORAGE_KEY);
  }
}

/**
 * Call DeepSeek AI to analyze the uploaded floor plan.
 * Handles HTTP 402 (Insufficient Balance) and network issues gracefully
 * so the application never breaks.
 */
export async function analyzeFloorPlanWithAI({
  base64Image,
  imageWidth,
  imageHeight,
  floorPlanScale = 100,
  apiKey = null
}) {
  const activeKey = apiKey || getDeepSeekApiKey();

  if (!activeKey) {
    return {
      success: false,
      isQuotaExceeded: false,
      error: "No DeepSeek API key provided.",
      message: "API key not configured. Using local computer vision NavMesh corridor engine."
    };
  }

  const promptText = `You are an autonomous drone tactical navigation security AI.
Analyze this factory floor plan (${imageWidth}x${imageHeight} px, real-world width: ${floorPlanScale}m).
1. Identify primary navigable air corridors and clear hallways.
2. Highlight high-risk obstacle clusters (heavy machinery, metallic racking).
3. Identify potential GPS/RF multipath reflection zones where spoofing signals might bounce.
4. Recommend optimal indoor flight corridors and emergency hold locations.
Provide a concise, tactical 3-sentence summary.`;

  try {
    const response = await fetch("https://api.deepseek.com/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${activeKey}`
      },
      body: JSON.stringify({
        model: "deepseek-v4-flash-vision-exp",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: promptText },
              ...(base64Image
                ? [{ type: "image_url", image_url: { url: base64Image } }]
                : [])
            ]
          }
        ],
        max_tokens: 300,
        temperature: 0.3
      })
    });

    if (response.status === 402) {
      // DeepSeek account balance exhausted
      return {
        success: false,
        isQuotaExceeded: true,
        error: "DeepSeek API: Insufficient Account Balance (HTTP 402)",
        message: `DeepSeek Vision (Key: sk-...${activeKey.slice(-4)}): Account balance exhausted. Activated High-Precision Local Computer Vision & NavMesh Corridor Engine.`
      };
    }

    if (!response.ok) {
      const errorText = await response.text();
      return {
        success: false,
        isQuotaExceeded: response.status === 429,
        error: `DeepSeek API Error (${response.status}): ${errorText}`,
        message: `DeepSeek Vision status ${response.status}. Fallback local NavMesh engine active.`
      };
    }

    const data = await response.json();
    const aiAnalysis = data.choices?.[0]?.message?.content?.trim() || "";

    return {
      success: true,
      isQuotaExceeded: false,
      message: aiAnalysis || "DeepSeek Vision: Navigable corridor analysis complete.",
      rawResponse: data
    };
  } catch (err) {
    return {
      success: false,
      isQuotaExceeded: false,
      error: err.message,
      message: "DeepSeek connection offline. Local computer vision engine operating at full capacity."
    };
  }
}
