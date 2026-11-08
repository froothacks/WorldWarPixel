import { useEffect, useState, type CSSProperties } from "react";

export default function Avatar({
  src,
  color,
  size,
}: {
  src?: string | null;
  color: string;
  size: number;
}) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);
  if (!src || failed) return <span className="swatch" style={{ background: color }} />;
  return (
    <img
      className="avatar"
      src={src}
      alt=""
      width={size}
      height={size}
      loading="lazy"
      referrerPolicy="no-referrer"
      style={{ "--c": color } as CSSProperties}
      onError={() => setFailed(true)}
    />
  );
}
