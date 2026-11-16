import { expect, test } from "bun:test";
import { humanBytes, num } from "./util";

test("humanBytes scales to human units", () => {
  expect(humanBytes(0)).toBe("0 B");
  expect(humanBytes(999)).toBe("999 B");
  expect(humanBytes(1024)).toBe("1.0 KiB");
  expect(humanBytes(1536)).toBe("1.5 KiB");
  expect(humanBytes(5 * 1024 ** 3)).toBe("5.0 GiB");
  expect(humanBytes(1024 ** 4 * 3)).toBe("3.0 TiB");
});

test("humanBytes tolerates junk sizes", () => {
  expect(humanBytes(-5)).toBe("0 B");
  expect(humanBytes(Number.NaN)).toBe("0 B");
});

test("num coerces generated bigint fields", () => {
  expect(num(42)).toBe(42);
  expect(num(BigInt(42))).toBe(42);
});
