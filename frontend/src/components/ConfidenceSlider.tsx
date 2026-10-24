import {
  CONFIDENCE_MAX,
  CONFIDENCE_MIN,
  CONFIDENCE_STEP,
} from "../lib/detections";

interface ConfidenceSliderProps {
  value: number;
  onChange: (value: number) => void;
  style?: React.CSSProperties;
}

/** Client-side confidence filter; the range matches what the server returns. */
export function ConfidenceSlider({ value, onChange, style }: ConfidenceSliderProps) {
  return (
    <input
      type="range"
      min={CONFIDENCE_MIN}
      max={CONFIDENCE_MAX}
      step={CONFIDENCE_STEP}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="threshold-slider"
      style={style}
    />
  );
}
