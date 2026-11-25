import { expect, test } from "bun:test";
import openapi from "./gen/openapi.json";

const doc = openapi as unknown as {
  components: { schemas: Record<string, Record<string, unknown>>; };
};

test("generated API artifacts expose the new image-ops contract", () => {
  const { schemas } = doc.components;
  expect(schemas.BlenderBakeMap!.enum).toContain("metallic");
  expect(JSON.stringify(schemas.BlenderStage)).toContain("thumbnail");
  expect(Object.keys(schemas.ImageOpOutput!.properties as object)).toContain(
    "artifact_id",
  );
});
